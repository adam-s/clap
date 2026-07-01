<script lang="ts">
  /**
   * ComputationInset — the "how stage k became stage k+1" explainer. A compact
   * canvas that animates the real operation feeding the active stage:
   *   mel     → filterbank: power spectrum · triangular filter → one mel bin
   *   conv    → a 3×3 kernel slides, multiply-add → one output cell
   *   pool    → a 2×2 patch averages → one pooled number
   *   embed   → matmul: a weight row · the 64-vector → one of 512
   * Numbers are real (kernels/weights captured), so the result reconciles with
   * the activations in the big view. 3b1b-style: one output cell at a time.
   */
  import { onMount, onDestroy } from 'svelte';
  import { clipIdx, stageIdx, STAGES } from '../stores/clap';
  import type { ClapBundle, TensorMap } from '../data/clapData';
  import { channel, channelMean } from '../data/clapData';
  import { diverging, sequential } from './colormap';

  type Props = { bundle: ClapBundle };
  let { bundle }: Props = $props();

  let ci = $state(0);
  let si = $state(0);
  const u1 = clipIdx.subscribe((v) => (ci = v));
  const u2 = stageIdx.subscribe((v) => (si = v));

  let canvas: HTMLCanvasElement;
  let wrap: HTMLDivElement;
  let cssW = $state(680);
  const H = 188;
  // The representative cell whose computation is shown. Static — the reader
  // scrolls at their own pace; nothing auto-cycles. Defaults to the most
  // active output cell so the example is informative.
  let sel = $state(0);

  // op type feeding stage `si`
  type Op = 'input' | 'filterbank' | 'conv' | 'pool' | 'matmul';
  const OP: Op[] = ['input', 'filterbank', 'conv', 'conv', 'conv', 'conv', 'pool', 'matmul'];
  const op = $derived(OP[si]);

  const FORMULA: Record<Op, string> = {
    input: 'x ∈ ℝ³²⁰⁰⁰ — raw audio samples',
    filterbank: 'mel[m] = Σ_f power[f] · melfb[f, m]   (triangular filters)',
    conv: 'Y[c,i,j] = b[c] + Σ_{c′,u,v} K[c,c′,u,v] · X[c′, 2i+u−1, 2j+v−1],  then ReLU',
    pool: 'z[c] = mean_{i,j} X[c,i,j]   (global average pool)',
    matmul: 'h[j] = b[j] + Σ_k W[j,k] · z[k]   (projection 64 → 512)',
  };
  const TITLE: Record<Op, string> = {
    input: 'Input — the waveform',
    filterbank: 'Waveform → Mel: the filterbank',
    conv: '3×3 convolution, stride 2',
    pool: 'Global average pool',
    matmul: 'Linear projection to 512',
  };

  // ── data plumbing per op ───────────────────────────────────────────────────
  const clip = $derived(bundle.clips[ci]);

  function inputMap(): TensorMap {
    if (si === 2) return clip.mel; // [32,32,1]
    if (si === 3) return clip.stages.patch;
    if (si === 4) return clip.stages.stage1;
    if (si === 5) return clip.stages.stage2;
    return clip.mel;
  }
  function outMap(): TensorMap {
    if (si === 2) return clip.stages.patch;
    if (si === 3) return clip.stages.stage1;
    if (si === 4) return clip.stages.stage2;
    return clip.stages.stage3;
  }
  function convKey() {
    return (['', '', 'patch', 'stage1', 'stage2', 'stage3'][si]) as
      'patch' | 'stage1' | 'stage2' | 'stage3';
  }

  // real full conv at output (oi,oj) for output channel 0
  function convCell(inp: TensorMap, K: TensorMap, b: TensorMap, oi: number, oj: number): number {
    const [H_, W_, Cin] = inp.shape;
    const [, , kh, kw] = K.shape;
    let s = b.data[0];
    for (let ic = 0; ic < Cin; ic++)
      for (let u = 0; u < kh; u++)
        for (let v = 0; v < kw; v++) {
          const ii = 2 * oi + u - 1;
          const jj = 2 * oj + v - 1;
          if (ii < 0 || jj < 0 || ii >= H_ || jj >= W_) continue;
          const x = inp.data[(ii * W_ + jj) * Cin + ic];
          const k = K.data[((0 * Cin + ic) * kh + u) * kw + v];
          s += x * k;
        }
    return Math.max(0, s);
  }

  function argmaxAbs(a: number[]): number {
    let bi = 0, bv = -1;
    for (let i = 0; i < a.length; i++) { const v = Math.abs(a[i]); if (v > bv) { bv = v; bi = i; } }
    return bi;
  }
  // representative cell index for the current op (most active → informative)
  function repIndex(): number {
    if (op === 'conv') return argmaxAbs(channel(outMap(), 0));
    if (op === 'pool') return argmaxAbs(clip.latent.data);
    if (op === 'matmul') return argmaxAbs(clip.audioEmbed);
    if (op === 'filterbank') {
      const col: number[] = [];
      for (let mm = 0; mm < bundle.meta.mel_bins; mm++) col.push(clip.mel.data[Math.min(mm, bundle.meta.img - 1)]);
      return argmaxAbs(col);
    }
    return 0;
  }

  // ── drawing ────────────────────────────────────────────────────────────────
  function cellRect(ctx: CanvasRenderingContext2D, x: number, y: number, s: number, color: string, outline = '') {
    ctx.fillStyle = color; ctx.fillRect(x, y, s, s);
    if (outline) { ctx.strokeStyle = outline; ctx.lineWidth = 1.5; ctx.strokeRect(x + 0.5, y + 0.5, s - 1, s - 1); }
  }
  function grid(ctx: CanvasRenderingContext2D, x: number, y: number, vals: number[], rows: number, cols: number, cell: number, amax: number, seq = false, lo = 0, hi = 1) {
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        const v = vals[r * cols + c];
        cellRect(ctx, x + c * cell, y + r * cell, cell - 1, seq ? sequential(v, lo, hi) : diverging(v, amax));
      }
  }
  function label(ctx: CanvasRenderingContext2D, t: string, x: number, y: number, color = '#8aa0b4') {
    ctx.fillStyle = color; ctx.font = '600 11px ui-monospace, Menlo, monospace'; ctx.textBaseline = 'alphabetic'; ctx.fillText(t, x, y);
  }

  function paint() {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = cssW * dpr; canvas.height = H * dpr;
    canvas.style.width = `${cssW}px`; canvas.style.height = `${H}px`;
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, H);

    if (op === 'conv') paintConv(ctx);
    else if (op === 'matmul') paintMatmul(ctx);
    else if (op === 'pool') paintPool(ctx);
    else if (op === 'filterbank') paintFilterbank(ctx);
    else paintInput(ctx);
  }

  function paintInput(ctx: CanvasRenderingContext2D) {
    label(ctx, 'The raw 16 kHz waveform — the input to the audio tower.', 16, H / 2);
  }

  function paintConv(ctx: CanvasRenderingContext2D) {
    const inp = inputMap(), out = outMap();
    const K = bundle.weights.conv[convKey()], b = bundle.weights.convBias[convKey()];
    const [Hi, Wi] = inp.shape;
    const [Ho, Wo] = out.shape;
    const total = Ho * Wo;
    const oi = Math.floor(sel / Wo) % Ho;
    const oj = sel % Wo;

    // input map (channel 0) with sliding 3×3 window
    const inCh = Wi <= 1 ? inp.data : channel(inp, 0);
    let iAmax = 0; for (const v of inCh) iAmax = Math.max(iAmax, Math.abs(v)); iAmax ||= 1e-6;
    const icell = Math.min(14, Math.floor(120 / Hi));
    const ix = 20, iy = 40;
    grid(ctx, ix, iy, inCh, Hi, Wi, icell, iAmax);
    label(ctx, `input X · ch0  (${Hi}×${Wi})`, ix, iy - 8);
    // window
    const wx = ix + (2 * oj - 1) * icell, wy = iy + (2 * oi - 1) * icell;
    ctx.strokeStyle = '#5aa9e6'; ctx.lineWidth = 2;
    ctx.strokeRect(wx, wy, 3 * icell, 3 * icell);

    // kernel 3×3 (outc0, inc0)
    const kcell = 22, kx = ix + Wi * icell + 46, ky = 52;
    let kAmax = 0; for (const v of K.data) kAmax = Math.max(kAmax, Math.abs(v)); kAmax ||= 1e-6;
    const ker: number[] = [];
    for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) ker.push(K.data[((0 * inp.shape[2] + 0) * 3 + u) * 3 + v]);
    grid(ctx, kx, ky, ker, 3, 3, kcell, kAmax);
    label(ctx, 'kernel K', kx, ky - 8, '#e6a15a');
    label(ctx, '✱', ix + Wi * icell + 22, ky + 1.5 * kcell, '#9aa7b4');

    // = output cell
    const val = convCell(inp, K, b, oi, oj);
    const ocx = kx + 3 * kcell + 40;
    label(ctx, '=', kx + 3 * kcell + 14, ky + 1.5 * kcell, '#9aa7b4');
    const ocell = 40;
    cellRect(ctx, ocx, ky + kcell, ocell, diverging(val, 1), '#5aa9e6');
    ctx.fillStyle = '#0e1116'; ctx.font = '600 12px ui-monospace,Menlo'; ctx.textBaseline = 'middle';
    ctx.textAlign = 'center'; ctx.fillText(val.toFixed(2), ocx + ocell / 2, ky + kcell + ocell / 2); ctx.textAlign = 'left';

    // output map filling in
    const omcell = Math.min(16, Math.floor(120 / Ho));
    const omx = ocx + ocell + 50, omy = 40;
    let oAmax = 0; const oc0 = channel(out, 0); for (const v of oc0) oAmax = Math.max(oAmax, Math.abs(v)); oAmax ||= 1e-6;
    for (let k = 0; k < total; k++) {
      const r = Math.floor(k / Wo), c = k % Wo;
      cellRect(ctx, omx + c * omcell, omy + r * omcell, omcell - 1,
        diverging(oc0[k], oAmax), k === sel ? '#5aa9e6' : '');
    }
    label(ctx, `output Y · ch0  (${Ho}×${Wo})`, omx, omy - 8);
  }

  function paintPool(ctx: CanvasRenderingContext2D) {
    const inp = clip.stages.stage3; // [2,2,64]
    const C = inp.shape[2];
    const c = sel % C;
    const patch = channel(inp, c);
    let amax = 0; for (const v of inp.data) amax = Math.max(amax, Math.abs(v)); amax ||= 1e-6;
    const cell = 34, x = 24, y = 56;
    grid(ctx, x, y, patch, 2, 2, cell, amax);
    label(ctx, `stage3 · ch${c}  (2×2)`, x, y - 8);
    const mean = patch.reduce((a, v) => a + v, 0) / patch.length;
    label(ctx, '→  mean  →', x + 2 * cell + 16, y + cell, '#9aa7b4');
    const ox = x + 2 * cell + 110;
    cellRect(ctx, ox, y + cell / 2, 36, diverging(mean, amax), '#5aa9e6');
    ctx.fillStyle = '#0e1116'; ctx.font = '600 12px ui-monospace,Menlo'; ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
    ctx.fillText(mean.toFixed(2), ox + 18, y + cell / 2 + 18); ctx.textAlign = 'left';

    // the 64-vector filling
    const vcell = 9, vx = ox + 90, vy = 40, cols = 16;
    const lat = clip.latent.data;
    let lAmax = 0; for (const v of lat) lAmax = Math.max(lAmax, Math.abs(v)); lAmax ||= 1e-6;
    for (let k = 0; k < C; k++) {
      const r = Math.floor(k / cols), cc = k % cols;
      cellRect(ctx, vx + cc * (vcell + 1), vy + r * (vcell + 1), vcell, diverging(lat[k], lAmax), k === c ? '#5aa9e6' : '');
    }
    label(ctx, 'pooled z (64)', vx, vy - 8);
  }

  function paintMatmul(ctx: CanvasRenderingContext2D) {
    const z = clip.latent.data;     // 64
    const W = bundle.weights.proj1; // [512,64]
    const b = bundle.weights.proj1Bias;
    const j = sel % 512;
    // latent column
    let zAmax = 0; for (const v of z) zAmax = Math.max(zAmax, Math.abs(v)); zAmax ||= 1e-6;
    const zc = 10, zx = 24, zy = 30, zcols = 8;
    for (let k = 0; k < 64; k++) {
      const r = Math.floor(k / zcols), c = k % zcols;
      cellRect(ctx, zx + c * (zc + 1), zy + r * (zc + 1), zc, diverging(z[k], zAmax));
    }
    label(ctx, 'z (64)', zx, zy - 8);
    // weight row j
    const wcols = 32, wc = 9, wx = zx + zcols * (zc + 1) + 40, wy = 30;
    let wAmax = 0; for (let k = 0; k < 64; k++) wAmax = Math.max(wAmax, Math.abs(W.data[j * 64 + k])); wAmax ||= 1e-6;
    const wrow: number[] = []; for (let k = 0; k < 64; k++) wrow.push(W.data[j * 64 + k]);
    grid(ctx, wx, wy, wrow, 2, 32, wc, wAmax);
    label(ctx, `W row j=${j}  (1×64)`, wx, wy - 8, '#e6a15a');
    // dot product = h[j]
    let dot = b.data[j]; for (let k = 0; k < 64; k++) dot += z[k] * W.data[j * 64 + k];
    const ox = wx + wcols * (wc + 1) + 30;
    label(ctx, '·  Σ  =', wx + wcols * (wc + 1) + 6, wy + wc, '#9aa7b4');
    cellRect(ctx, ox, wy, 40, diverging(dot, 4), '#5aa9e6');
    ctx.fillStyle = '#0e1116'; ctx.font = '600 12px ui-monospace,Menlo'; ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
    ctx.fillText(dot.toFixed(2), ox + 20, wy + 20); ctx.textAlign = 'left';
    label(ctx, `→ one of the 512 pre-activations (dim j = ${j}); repeat 512× for the full embedding shown below.`, 24, 150, '#6b7785');
  }

  function paintFilterbank(ctx: CanvasRenderingContext2D) {
    const fb = bundle.weights.melfb; // [n_freqs, n_mels]
    const [nf, nm] = fb.shape;
    const m = sel % nm;
    // plot triangular filter m over freq axis
    const x0 = 24, y0 = 110, plotW = Math.min(cssW - 240, 420), plotH = 70;
    ctx.strokeStyle = 'var(--line)'; ctx.strokeStyle = '#2a323d';
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + plotW, y0); ctx.stroke();
    // a few neighbor filters faint, active one bright
    for (let mm = Math.max(0, m - 2); mm <= Math.min(nm - 1, m + 2); mm++) {
      ctx.strokeStyle = mm === m ? '#5aa9e6' : 'rgba(120,135,150,0.3)';
      ctx.lineWidth = mm === m ? 2 : 1;
      ctx.beginPath();
      for (let f = 0; f < nf; f++) {
        const w = fb.data[f * nm + mm];
        const x = x0 + (f / (nf - 1)) * plotW;
        const y = y0 - w * plotH * 6;
        f === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    label(ctx, `mel filter m=${m} of ${nm}  (triangular weights over ${nf} freq bins)`, x0, y0 - plotH - 6);
    label(ctx, 'frequency →', x0 + plotW - 70, y0 + 14, '#6b7785');
    // resulting mel column (real, first time frame)
    const col: number[] = [];
    for (let mm = 0; mm < nm; mm++) col.push(clip.mel.data[(0 * bundle.meta.img + Math.min(mm, bundle.meta.img - 1)) * 1]);
    const vx = x0 + plotW + 60, vy = 24, vc = 5;
    let lo = Infinity, hi = -Infinity; for (const v of col) { if (v < lo) lo = v; if (v > hi) hi = v; }
    for (let mm = 0; mm < nm; mm++)
      cellRect(ctx, vx, vy + mm * vc, vc * 2, sequential(col[mm], lo, hi), mm === m ? '#5aa9e6' : '');
    label(ctx, `mel column (${nm})`, vx - 4, vy - 8);
  }

  // Static: recompute the representative cell and repaint only when the clip,
  // stage, or width changes. No auto-cycling — the reader scrolls at their pace.
  $effect(() => { void [ci, si, cssW]; sel = repIndex(); paint(); });

  onMount(() => {
    const ro = new ResizeObserver((e) => (cssW = e[0].contentRect.width));
    ro.observe(wrap);
    return () => ro.disconnect();
  });
  onDestroy(() => { u1(); u2(); });
</script>

<div bind:this={wrap} class="inset">
  <div class="hdr">
    <span class="step mono">Step {si} / {STAGES.length - 1}</span>
    <span class="title">{TITLE[op]}</span>
  </div>
  <div class="formula mono">{FORMULA[op]}</div>
  <canvas bind:this={canvas}></canvas>
</div>

<style>
  .inset {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: var(--space-md);
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .hdr { display: flex; align-items: baseline; gap: 12px; }
  .step { color: var(--accent); font-size: 0.74rem; }
  .title { font-weight: 600; font-size: 0.96rem; }
  .formula {
    color: var(--ink-dim);
    font-size: 0.78rem;
    background: var(--panel-2);
    padding: 6px 10px;
    border-radius: 6px;
    overflow-x: auto;
    white-space: nowrap;
  }
  canvas { display: block; width: 100%; }
</style>
