<script lang="ts">
  /**
   * PipelineCanvas — the whole audio tower unfolded on ONE full-bleed canvas.
   * Every stage is drawn in full (waveform → 32×32 mel → patch grid → the three
   * shrinking hierarchical maps, all channels → pooled 64 → 512 embedding), and
   * elegant bezier strands connect each stage's OUTPUT extent to the next
   * stage's INPUT extent, so the representation widening/collapsing is visible.
   * Dashes travel output→input; the strand feeding the active stage animates.
   *
   * Canvas pattern follows grpo's ModelDiagram: DPR-sized 2D canvas, direct
   * cell fills, a static offscreen layer + a cheap animated overlay each frame.
   */
  import { onDestroy, onMount } from 'svelte';
  import { clipIdx, stageIdx, STAGES, setStage, playing } from '../stores/clap';
  import type { ClapBundle, TensorMap } from '../data/clapData';
  import { diverging, sequential } from './colormap';

  type Props = { bundle: ClapBundle };
  let { bundle }: Props = $props();

  let wrap: HTMLDivElement;
  let canvas: HTMLCanvasElement;
  let off: HTMLCanvasElement | null = null;

  let ci = 0;
  let si = 0;
  let isPlaying = false;

  let cssW = 0;
  let cssH = 0;
  let phase = 0;
  let raf = 0;
  let visible = true;

  // ── Layout model ──────────────────────────────────────────────────────────
  type Block = {
    key: string;
    label: string;
    sub: string;
    yTop: number;     // top of the drawn tensor (below its label)
    yBot: number;     // bottom of the drawn tensor
    xL: number;       // left of the drawn extent
    xR: number;       // right of the drawn extent
    blockTop: number; // top including label (for hit-test + highlight)
    blockBot: number;
    draw: (ctx: CanvasRenderingContext2D) => void;
  };
  let blocks: Block[] = [];

  // scroll ↔ stage sync state
  let ready = false;
  let scrollDriven = false;
  let suppressUntil = 0;

  const subs = [
    clipIdx.subscribe((v) => { ci = v; rebuild(); }),
    stageIdx.subscribe((v) => { si = v; rebuild(); if (ready && !scrollDriven) scrollToStage(v); }),
    playing.subscribe((v) => (isPlaying = v)),
  ];

  const PAD = 18;
  const LABEL_H = 26;
  const GAP = 64; // vertical room for connectors

  function drawWidth() {
    return Math.min(cssW - PAD * 2, 1080);
  }

  // generic small-multiples layout for a [H,W,C] map
  function chanLayout(H: number, W: number, C: number, cellPx: number, chanGap: number, maxW: number) {
    const bw = W * cellPx;
    const bh = H * cellPx;
    const perRow = Math.max(1, Math.min(C, Math.floor((maxW + chanGap) / (bw + chanGap))));
    const rows = Math.ceil(C / perRow);
    return {
      perRow, rows, bw, bh,
      totalW: perRow * bw + (perRow - 1) * chanGap,
      totalH: rows * bh + (rows - 1) * chanGap,
    };
  }

  function absMaxOf(m: TensorMap) {
    let a = 0;
    for (const v of m.data) a = Math.max(a, Math.abs(v));
    return a || 1e-6;
  }

  function fillCell(ctx: CanvasRenderingContext2D, x: number, y: number, s: number, color: string) {
    ctx.fillStyle = color;
    ctx.fillRect(x, y, s, s);
  }

  // draw one [H,W,C] map as a row/grid of per-channel heatmaps; returns extent
  function drawChannels(
    ctx: CanvasRenderingContext2D, cx: number, top: number, m: TensorMap,
    cellPx: number, chanGap: number,
  ) {
    const [H, W, C] = m.shape;
    const L = chanLayout(H, W, C, cellPx, chanGap, drawWidth());
    const amax = absMaxOf(m);
    const x0 = cx - L.totalW / 2;
    for (let c = 0; c < C; c++) {
      const r = Math.floor(c / L.perRow);
      const col = c % L.perRow;
      const bx = x0 + col * (L.bw + chanGap);
      const by = top + r * (L.bh + chanGap);
      ctx.fillStyle = 'rgba(255,255,255,0.04)';
      ctx.fillRect(bx - 1, by - 1, L.bw + 2, L.bh + 2);
      for (let i = 0; i < H; i++)
        for (let j = 0; j < W; j++)
          fillCell(ctx, bx + j * cellPx, by + i * cellPx, cellPx, diverging(m.data[(i * W + j) * C + c], amax));
    }
    return { w: L.totalW, h: L.totalH, xL: x0, xR: x0 + L.totalW };
  }

  function drawGridSeq(ctx: CanvasRenderingContext2D, cx: number, top: number, vals: number[], n: number, cell: number) {
    let lo = Infinity, hi = -Infinity;
    for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const w = n * cell;
    const x0 = cx - w / 2;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        fillCell(ctx, x0 + j * cell, top + i * cell, cell, sequential(vals[i * n + j], lo, hi));
    return { w, h: n * cell, xL: x0, xR: x0 + w };
  }

  function drawVector(ctx: CanvasRenderingContext2D, cx: number, top: number, vals: number[], cols: number, cell: number, gap: number) {
    let amax = 0;
    for (const v of vals) amax = Math.max(amax, Math.abs(v));
    amax = amax || 1e-6;
    const rows = Math.ceil(vals.length / cols);
    const w = cols * (cell + gap) - gap;
    const x0 = cx - w / 2;
    for (let k = 0; k < vals.length; k++) {
      const r = Math.floor(k / cols), c = k % cols;
      fillCell(ctx, x0 + c * (cell + gap), top + r * (cell + gap), cell, diverging(vals[k], amax));
    }
    return { w, h: rows * (cell + gap) - gap, xL: x0, xR: x0 + w };
  }

  function build() {
    const clip = bundle.clips[ci];
    const cx = cssW / 2;
    const W = drawWidth();
    blocks = [];
    let y = PAD;

    const add = (
      key: string, label: string, sub: string,
      measureH: number,
      paint: (ctx: CanvasRenderingContext2D, top: number) => { xL: number; xR: number },
    ) => {
      const blockTop = y;
      const yTop = y + LABEL_H;
      let extent = { xL: cx - 10, xR: cx + 10 };
      const draw = (ctx: CanvasRenderingContext2D) => {
        ctx.fillStyle = '#8aa0b4';
        ctx.font = '600 12px ui-monospace, Menlo, monospace';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${label}`, cx - W / 2, blockTop + LABEL_H / 2);
        ctx.fillStyle = '#5a6675';
        ctx.fillText(sub, cx - W / 2 + ctx.measureText(label).width + 12, blockTop + LABEL_H / 2);
        extent = paint(ctx, yTop);
      };
      const b: Block = {
        key, label, sub, yTop, yBot: yTop + measureH,
        xL: cx - 10, xR: cx + 10, blockTop, blockBot: yTop + measureH, draw,
        // extent resolved at draw time; store getter via closure
      };
      // patch extent after first draw by re-reading; simpler: compute in paint and stash
      (b as any)._resolve = () => { b.xL = extent.xL; b.xR = extent.xR; };
      blocks.push(b);
      y = yTop + measureH + GAP;
    };

    // 1 waveform
    add('waveform', 'Waveform', STAGES[0].sub, 84, (ctx, top) => {
      const w = W * 0.94;
      const x0 = cx - w / 2;
      const h = 84;
      const mid = top + h / 2;
      let peak = 0;
      for (const s of clip.waveform) peak = Math.max(peak, Math.abs(s));
      const norm = peak > 0 ? (h / 2) / peak : 0;
      ctx.strokeStyle = 'rgba(90,169,230,0.25)';
      ctx.beginPath(); ctx.moveTo(x0, mid); ctx.lineTo(x0 + w, mid); ctx.stroke();
      ctx.strokeStyle = '#5aa9e6';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < clip.waveform.length; i++) {
        const x = x0 + (i / (clip.waveform.length - 1)) * w;
        const yy = mid - clip.waveform[i] * norm * 0.92;
        i ? ctx.lineTo(x, yy) : ctx.moveTo(x, yy);
      }
      ctx.stroke();
      return { xL: x0, xR: x0 + w };
    });

    // 2 mel
    const melCell = 8;
    add('mel', 'Mel image', STAGES[1].sub, bundle.meta.img * melCell, (ctx, top) =>
      drawGridSeq(ctx, cx, top, clip.mel.data, bundle.meta.img, melCell));

    // 3 patch
    const pL = chanLayout(16, 16, 8, 5, 8, W);
    add('patch', 'Patch embed', STAGES[2].sub, pL.totalH, (ctx, top) =>
      drawChannels(ctx, cx, top, clip.stages.patch, 5, 8));

    // 4 stage1
    const s1 = chanLayout(8, 8, 16, 7, 6, W);
    add('stage1', 'Stage 1', STAGES[3].sub, s1.totalH, (ctx, top) =>
      drawChannels(ctx, cx, top, clip.stages.stage1, 7, 6));

    // 5 stage2
    const s2 = chanLayout(4, 4, 32, 9, 5, W);
    add('stage2', 'Stage 2', STAGES[4].sub, s2.totalH, (ctx, top) =>
      drawChannels(ctx, cx, top, clip.stages.stage2, 9, 5));

    // 6 stage3
    const s3 = chanLayout(2, 2, 64, 11, 4, W);
    add('stage3', 'Stage 3', STAGES[5].sub, s3.totalH, (ctx, top) =>
      drawChannels(ctx, cx, top, clip.stages.stage3, 11, 4));

    // 7 latent
    const latCols = Math.min(64, Math.floor(W / 15));
    add('latent', 'Pooled', STAGES[6].sub, Math.ceil(64 / latCols) * 15, (ctx, top) =>
      drawVector(ctx, cx, top, clip.latent.data, latCols, 13, 2));

    // 8 embed
    const embCols = Math.min(64, Math.floor(W / 12));
    add('embed', 'Embedding', STAGES[7].sub, Math.ceil(512 / embCols) * 12, (ctx, top) =>
      drawVector(ctx, cx, top, clip.audioEmbed, embCols, 10, 2));

    cssH = y - GAP + PAD;
  }

  // ── Render ────────────────────────────────────────────────────────────────
  function paintStatic() {
    if (!off) off = document.createElement('canvas');
    const dpr = window.devicePixelRatio || 1;
    off.width = cssW * dpr;
    off.height = cssH * dpr;
    const ctx = off.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    blocks.forEach((b, i) => {
      if (i === si) {
        ctx.fillStyle = 'rgba(90,169,230,0.07)';
        const x = cssW / 2 - drawWidth() / 2 - 10;
        roundRect(ctx, x, b.blockTop - 6, drawWidth() + 20, b.blockBot - b.blockTop + 12, 8);
        ctx.fill();
        ctx.fillStyle = '#5aa9e6';
        ctx.fillRect(x, b.blockTop - 6, 3, b.blockBot - b.blockTop + 12);
      }
      b.draw(ctx);
      (b as any)._resolve?.();
    });
  }

  function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  const STRANDS: number = 5;

  function paintFrame() {
    if (!canvas || !off) return;
    const dpr = window.devicePixelRatio || 1;
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.drawImage(off, 0, 0, cssW, cssH);

    for (let k = 0; k < blocks.length - 1; k++) {
      const a = blocks[k];
      const b = blocks[k + 1];
      const state = k + 1 === si ? 'flow' : k + 1 < si ? 'done' : 'idle';
      drawConnector(ctx, a, b, state);
    }
  }

  function drawConnector(ctx: CanvasRenderingContext2D, a: Block, b: Block, state: 'idle' | 'flow' | 'done') {
    const color = state === 'flow' ? '#5aa9e6' : state === 'done' ? 'rgba(90,169,230,0.45)' : 'rgba(120,135,150,0.28)';
    ctx.strokeStyle = color;
    ctx.lineWidth = state === 'flow' ? 1.6 : 1;
    const y1 = a.yBot;
    const y2 = b.yTop;
    const dash = state === 'flow';
    if (dash) { ctx.setLineDash([3, 7]); ctx.lineDashOffset = -phase; }
    else ctx.setLineDash([]);
    for (let s = 0; s < STRANDS; s++) {
      const t = STRANDS === 1 ? 0.5 : s / (STRANDS - 1);
      const x1 = a.xL + (a.xR - a.xL) * t;
      const x2 = b.xL + (b.xR - b.xL) * t;
      const cy = (y1 + y2) / 2;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.bezierCurveTo(x1, cy, x2, cy, x2, y2);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    // arrowhead into input
    const xm = (b.xL + b.xR) / 2;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(xm - 4, y2 - 6);
    ctx.lineTo(xm, y2 - 1);
    ctx.lineTo(xm + 4, y2 - 6);
    ctx.closePath();
    ctx.fill();
  }

  function rebuild() {
    if (!cssW) return;
    build();
    sizeCanvas();
    paintStatic();
    paintFrame();
  }

  function sizeCanvas() {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
  }

  function loop() {
    if (visible) {
      phase = (phase + 0.6) % 1000;
      paintFrame();
    }
    raf = requestAnimationFrame(loop);
  }

  // page-Y of a block's vertical center
  function pageYCenter(b: Block): number {
    const top = canvas.getBoundingClientRect().top + window.scrollY;
    return top + (b.blockTop + b.blockBot) / 2;
  }
  function scrollToStage(i: number) {
    if (!canvas || !blocks[i]) return;
    const target = pageYCenter(blocks[i]) - window.innerHeight / 2 + 80;
    suppressUntil = performance.now() + 800;
    window.scrollTo({ top: Math.max(0, target), behavior: 'smooth' });
  }
  let scrollRaf = 0;
  function onScroll() {
    if (scrollRaf) return;
    scrollRaf = requestAnimationFrame(() => {
      scrollRaf = 0;
      if (!ready || !blocks.length || performance.now() < suppressUntil) return;
      // only sync while the figure is actually on screen
      const rect = canvas.getBoundingClientRect();
      if (rect.bottom < 0 || rect.top > window.innerHeight) return;
      const mid = window.scrollY + window.innerHeight / 2;
      let best = si, bestD = Infinity;
      for (let i = 0; i < blocks.length; i++) {
        const d = Math.abs(pageYCenter(blocks[i]) - mid);
        if (d < bestD) { bestD = d; best = i; }
      }
      if (best !== si) { scrollDriven = true; setStage(best); scrollDriven = false; }
    });
  }

  function onClick(e: MouseEvent) {
    const rect = canvas.getBoundingClientRect();
    const y = e.clientY - rect.top;
    for (let i = 0; i < blocks.length; i++) {
      if (y >= blocks[i].blockTop - 6 && y <= blocks[i].blockBot + GAP / 2) {
        playing.set(false);
        setStage(i);
        return;
      }
    }
  }

  onMount(() => {
    const ro = new ResizeObserver((entries) => {
      cssW = entries[0].contentRect.width;
      rebuild();
      ready = true;
    });
    ro.observe(wrap);
    const io = new IntersectionObserver((e) => (visible = e[0].isIntersecting), { threshold: 0 });
    io.observe(canvas);
    window.addEventListener('scroll', onScroll, { passive: true });
    raf = requestAnimationFrame(loop);
    return () => { ro.disconnect(); io.disconnect(); window.removeEventListener('scroll', onScroll); };
  });

  onDestroy(() => {
    cancelAnimationFrame(raf);
    if (scrollRaf) cancelAnimationFrame(scrollRaf);
    subs.forEach((u) => u());
  });
</script>

<div bind:this={wrap} class="bleed">
  <canvas bind:this={canvas} onclick={onClick}></canvas>
</div>

<style>
  .bleed {
    width: 100vw;
    position: relative;
    left: 50%;
    transform: translateX(-50%);
  }
  canvas { display: block; margin: 0 auto; cursor: pointer; }
</style>
