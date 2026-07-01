<script lang="ts">
  /**
   * ConvStep — one conv stage as a self-contained, independently-played
   * breakout, in GRPO's idiom: airy (no boxed card), a centered KaTeX formula,
   * an "OpGrid" pair (input map with the sliding 3×3 window → output map with
   * the cell it produces), and a FocusDemo computation band — labelled operands
   * → the dot product building term-by-term → the result landing in a named
   * target cell that feeds the next step.
   */
  import { onDestroy, onMount } from 'svelte';
  import type { ClapBundle, Clip, TensorMap } from '../data/clapData';
  import { channel } from '../data/clapData';
  import { diverging, sequential, textColor } from './colormap';
  import KaTeX from '../components/Math.svelte';

  type StageKey = 'patch' | 'stage1' | 'stage2' | 'stage3';
  type Props = { bundle: ClapBundle; clip: Clip; stage: StageKey; n: number };
  let { bundle, clip, stage, n }: Props = $props();

  const FROM: Record<StageKey, string> = { patch: 'Mel image', stage1: 'Patch', stage2: 'Stage 1', stage3: 'Stage 2' };
  const TO: Record<StageKey, string> = { patch: 'Patch', stage1: 'Stage 1', stage2: 'Stage 2', stage3: 'Stage 3' };
  const NEXT: Record<StageKey, string> = { patch: 'Stage 1', stage1: 'Stage 2', stage2: 'Stage 3', stage3: 'the pool' };

  const inputMap = $derived<TensorMap>(
    stage === 'patch' ? clip.mel
    : stage === 'stage1' ? clip.stages.patch
    : stage === 'stage2' ? clip.stages.stage1
    : clip.stages.stage2,
  );
  const outMap = $derived<TensorMap>(clip.stages[stage]);
  const kernel = $derived(bundle.weights.conv[stage]);
  const bias = $derived(bundle.weights.convBias[stage]);

  const Hi = $derived(inputMap.shape[0]);
  const Wi = $derived(inputMap.shape[1]);
  const Cin = $derived(inputMap.shape[2]);
  const Ho = $derived(outMap.shape[0]);
  const Wo = $derived(outMap.shape[1]);
  const Cout = $derived(outMap.shape[2]);

  // Channel 0 is a dead filter for some clips (ReLU kills it everywhere), so
  // show the most-active output channel instead — and a live input channel.
  function activeChannel(m: TensorMap): number {
    const [H, W, C] = m.shape;
    let best = 0, bv = -1;
    for (let c = 0; c < C; c++) {
      let mx = 0;
      for (let i = 0; i < H * W; i++) { const v = Math.abs(m.data[i * C + c]); if (v > mx) mx = v; }
      if (mx > bv) { bv = mx; best = c; }
    }
    return best;
  }
  const oc = $derived(activeChannel(outMap));            // output channel shown
  const ic = $derived(Cin === 1 ? 0 : activeChannel(inputMap)); // input channel shown

  const inCh0 = $derived(Cin === 1 ? inputMap.data : channel(inputMap, ic));
  const inLo = $derived(Math.min(...inCh0));
  const inHi = $derived(Math.max(...inCh0));
  const outCh0 = $derived(channel(outMap, oc));
  const outLo = $derived(Math.min(...outCh0));
  const outHi = $derived(Math.max(...outCh0));
  const ker = $derived.by(() => {
    const k: number[] = [];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) k.push(kernel.data[((oc * Cin + ic) * 3 + a) * 3 + b]);
    return k;
  });
  const kerAbs = $derived(Math.max(1e-6, ...ker.map(Math.abs)));

  // ── per-instance animation: window pos + which of 9 taps ────────────────────
  let pos = $state(0);
  let tap = $state(9);
  let playing = $state(true);
  let visible = $state(true);     // only animate while on-screen (avoids starving other rAF loops)
  let root: HTMLElement;
  const oi = $derived(Math.floor(pos / Wo) % Ho);
  const oj = $derived(pos % Wo);

  const patch = $derived.by(() => {
    const p: number[] = [];
    for (let a = 0; a < 3; a++)
      for (let b = 0; b < 3; b++) {
        const ii = 2 * oi + a - 1, jj = 2 * oj + b - 1;
        p.push(ii < 0 || jj < 0 || ii >= Hi || jj >= Wi ? 0 : inCh0[ii * Wi + jj]);
      }
    return p;
  });
  const ch0sum = $derived.by(() => {
    let s = bias.data[oc];
    for (let t = 0; t < Math.min(tap, 9); t++) s += patch[t] * ker[t];
    return s;
  });
  const fullOut = $derived(outCh0[pos] ?? 0);
  const fmt = (x: number, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '—');

  let acc = 0, lastT = 0, raf = 0;
  function loop(t: number) {
    raf = requestAnimationFrame(loop);
    if (!playing || !visible) { lastT = t; return; }
    if (!lastT) lastT = t;
    acc += t - lastT; lastT = t;
    const need = tap < 9 ? 230 : 900;
    if (acc >= need) { acc = 0; if (tap < 9) tap += 1; else { pos = (pos + 1) % (Ho * Wo); tap = 0; } }
  }
  $effect(() => { void [clip, stage]; pos = 0; tap = 0; acc = 0; });
  onMount(() => {
    const io = new IntersectionObserver((e) => (visible = e[0].isIntersecting), { threshold: 0 });
    if (root) io.observe(root);
    raf = requestAnimationFrame(loop);
    return () => io.disconnect();
  });
  onDestroy(() => { cancelAnimationFrame(raf); });

  function inspect(k: number) { playing = false; pos = k; tap = 9; }

  const inCell = $derived(Math.max(5, Math.min(14, Math.floor(150 / Hi))));
  const outCell = $derived(Math.max(8, Math.min(30, Math.floor(150 / Ho))));
  const PK = 30;

  const tex = String.raw`\htmlClass{part-y}{Y_{i,j}} = \operatorname{ReLU}\!\Big(\htmlClass{part-b}{b} + \textstyle\sum_{u,v} \htmlClass{part-k}{K_{u,v}}\, \htmlClass{part-x}{X_{2i+u,\,2j+v}}\Big)`;

  // Per-stage explanation — each builds on the last, climbing the abstraction
  // ladder (detect patterns, then shrink). Define terms where first needed.
  const EXPLAIN: Record<StageKey, string> = {
    patch:
      `We begin with the mel image: one brightness value per pixel — how loud each ` +
      `frequency is at each moment. This first convolution slides <em>8 little 3×3 ` +
      `stencils</em> across it. Each stencil lights up on a simple local shape: a ` +
      `horizontal streak (a held tone), a vertical streak (a sharp click), an edge, ` +
      `a blob. So every region's single brightness is replaced by 8 numbers: how ` +
      `much each shape is present. We trade resolution (32→16) for meaning (1→8).`,
    stage1:
      `The input is no longer brightness — it's the 8 shape-scores from step 02. This ` +
      `convolution looks at 3×3 neighbourhoods of <em>those</em> and learns 16 ` +
      `combinations: "a click on top of a tone", "two streaks side by side". ` +
      `Patterns of patterns: that's how simple parts become complex ones. Each output ` +
      `cell now stands for a 4×4 chunk of the original spectrogram.`,
    stage2:
      `Same move, one rung higher. 32 detectors fuse the 16 mid-level features into ` +
      `bigger, more specific structures — closer to "what a whole knock or whine looks ` +
      `like". Each output cell now summarises an 8×8 slab of the original image: a ` +
      `sizeable stretch of the sound.`,
    stage3:
      `The last convolution. 64 detectors over a tiny 2×2 grid — each of these 4 cells ` +
      `describes a <em>quarter of the entire clip</em> with 64 high-level numbers. We've ` +
      `climbed from "brightness per pixel" to "a few rich descriptions of the whole ` +
      `sound", nearly ready to collapse into one vector.`,
  };
</script>

<section class="step" bind:this={root}>
  <header>
    <span class="eyebrow mono">{String(n).padStart(2, '0')} · {FROM[stage]} → {TO[stage]}</span>
    <span class="dims mono">{Hi}×{Wi}×{Cin} → {Ho}×{Wo}×{Cout}</span>
  </header>

  <p class="explain">{@html EXPLAIN[stage]}</p>

  <div class="formula"><KaTeX {tex} displayMode /></div>

  <!-- OpGrids: input (window) → output (cell) -->
  <div class="grids-row">
    {#if stage === 'patch'}
      <aside class="pad-note">
        the pale border is <em>zero-padding</em> — a ring of zeros so the 3×3
        window can still sit on the edge pixels, keeping the output a clean
        half-size
      </aside>
    {/if}
    <figure>
      <figcaption class="mono x">input X · ch{ic} <span class="dim">{Hi}×{Wi}{Cin > 1 ? ` of ${Cin}` : ''}</span></figcaption>
      <svg class="map" viewBox={`${-inCell} ${-inCell} ${(Wi + 2) * inCell} ${(Hi + 2) * inCell}`} width={(Wi + 2) * inCell} height={(Hi + 2) * inCell}>
        <rect class="pad" x={-inCell} y={-inCell} width={(Wi + 2) * inCell} height={(Hi + 2) * inCell} />
        {#each inCh0 as v, k (k)}
          <rect x={(k % Wi) * inCell} y={Math.floor(k / Wi) * inCell} width={inCell} height={inCell} fill={sequential(v, inLo, inHi)} />
        {/each}
        <rect class="win" x={(2 * oj - 1) * inCell} y={(2 * oi - 1) * inCell} width={3 * inCell} height={3 * inCell} />
      </svg>
    </figure>
    <span class="flow mono">→ Y[{oi},{oj}] →</span>
    <figure>
      <figcaption class="mono y">output Y · ch{oc} <span class="dim">{Ho}×{Wo}</span></figcaption>
      <svg class="map out" viewBox={`0 0 ${Wo * outCell} ${Ho * outCell}`} width={Wo * outCell} height={Ho * outCell}
        role="group" aria-label="output cells — hover to inspect" onpointerleave={() => (playing = true)}>
        {#each outCh0 as v, k (k)}
          <rect x={(k % Wo) * outCell} y={Math.floor(k / Wo) * outCell} width={outCell} height={outCell} fill={sequential(v, outLo, outHi)}
            role="button" tabindex="-1" onpointerenter={() => inspect(k)} />
        {/each}
        <rect class="cur-out" x={oj * outCell} y={oi * outCell} width={outCell} height={outCell} />
      </svg>
    </figure>
  </div>

  <!-- FocusDemo band: operands | equation building | lands in -->
  <div class="focus">
    <div class="operands">
      <div class="g">
        <span class="lab mono x">patch X</span>
        <svg viewBox={`0 0 ${3 * PK} ${3 * PK}`} width={3 * PK} height={3 * PK}>
          {#each patch as v, k (k)}
            {@const bg = sequential(v, inLo, inHi)}
            <rect x={(k % 3) * PK} y={Math.floor(k / 3) * PK} width={PK} height={PK} fill={bg} class:lit={k === tap} class:dim={tap < 9 && k > tap} />
            <text x={(k % 3) * PK + PK / 2} y={Math.floor(k / 3) * PK + PK / 2} class="ct" fill={textColor(bg)}>{fmt(v)}</text>
          {/each}
        </svg>
      </div>
      <span class="odot">⊙</span>
      <div class="g">
        <span class="lab mono k">kernel K</span>
        <svg viewBox={`0 0 ${3 * PK} ${3 * PK}`} width={3 * PK} height={3 * PK}>
          {#each ker as v, k (k)}
            {@const bg = diverging(v, kerAbs)}
            <rect x={(k % 3) * PK} y={Math.floor(k / 3) * PK} width={PK} height={PK} fill={bg} class:lit={k === tap} class:dim={tap < 9 && k > tap} />
            <text x={(k % 3) * PK + PK / 2} y={Math.floor(k / 3) * PK + PK / 2} class="ct" fill={textColor(bg)}>{fmt(v)}</text>
          {/each}
        </svg>
      </div>
    </div>

    <div class="eq">
      <div class="eq-title">Y[{oi},{oj}]</div>
      <div class="eq-line">= ReLU( b + Σ Xₖ·Kₖ )</div>
      <!-- all 9 terms always rendered (only their styling changes) so the line
           never grows / re-wraps and the layout stays put -->
      <div class="eq-line inter">
        = ReLU( {fmt(bias.data[oc])}{#each patch as x, k (k)}{' '}<span
          class="term" class:done={k < tap} class:hot={k === tap - 1} class:future={tap < 9 && k >= tap}
        >+ {fmt(x)}·{fmt(ker[k])}</span>{/each} )
      </div>
      <div class="eq-line result" class:on={tap >= 9}>
        = ReLU( {fmt(ch0sum)} ) = {fmt(Math.max(0, ch0sum))}
      </div>
    </div>

    <div class="target">
      <span class="t-lab mono">lands in</span>
      <span class="t-cell mono">{TO[stage]}[{oi},{oj}]</span>
      <span class="t-val mono">{fmt(fullOut)}</span>
      <span class="t-next mono">→ feeds {NEXT[stage]}{Cin > 1 ? ` · (Σ ${Cin} ch)` : ''}</span>
    </div>
  </div>

</section>

<style>
  /* Airy, GRPO-style: no boxed card. A hairline rule separates steps; the
     figure breathes in whitespace instead of sitting in a panel. */
  .step {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: var(--space-xl) 0;
    border-top: 1px solid var(--line);
  }
  header { display: flex; align-items: baseline; gap: 14px; }
  .eyebrow { color: var(--ink-faint); font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .dims { color: var(--ink-faint); font-size: 0.72rem; }
  .dims { margin-left: auto; }
  .explain { margin: 0; color: var(--ink-dim); font-size: 0.95rem; line-height: 1.6; max-width: 64ch; }
  .explain :global(em) { color: var(--ink); font-style: italic; }

  /* symbol ↔ picture color coding (X input, K kernel, Y output) */
  .x, .focus :global(.part-x) { color: var(--x-color); }
  .k, .focus :global(.part-k) { color: var(--k-color); }
  .y, .formula :global(.part-y) { color: var(--y-color); }
  .formula :global(.part-x) { color: var(--x-color); }
  .formula :global(.part-k) { color: var(--k-color); }
  .formula :global(.part-b) { color: var(--weight); }
  .formula { color: var(--ink); font-size: 1rem; margin: 2px 0; }

  .grids-row { position: relative; display: flex; align-items: center; justify-content: center; gap: var(--space-lg); flex-wrap: wrap; }
  figure { margin: 0; display: flex; flex-direction: column; align-items: center; gap: 6px; }
  figcaption, .lab { font-size: 0.7rem; color: var(--ink-dim); }
  .dim { color: var(--ink-faint); }
  .map { border-radius: 3px; image-rendering: pixelated; overflow: visible; }
  /* the 1-cell margin around the grid is the conv's zero-padding region */
  .pad { fill: var(--panel); }
  /* margin note in the left gutter — adds rhythm without touching the maps' alignment */
  .pad-note {
    position: absolute; left: 0; top: 50%; transform: translateY(-50%);
    width: 160px; text-align: right;
    font-size: 0.72rem; line-height: 1.45; font-style: italic;
    color: var(--ink-faint); pointer-events: none;
  }
  .pad-note em { color: var(--ink-dim); font-style: normal; }
  @media (max-width: 860px) { .pad-note { display: none; } }
  .win { fill: rgba(90,169,230,0.12); stroke: var(--accent); stroke-width: 2; transition: x 200ms ease, y 200ms ease; pointer-events: none; }
  .out rect { cursor: pointer; }
  .cur-out { fill: none; stroke: var(--accent); stroke-width: 2.5; pointer-events: none; transition: x 200ms ease, y 200ms ease; }
  .flow { color: var(--ink-faint); font-size: 0.78rem; white-space: nowrap; }

  /* FocusDemo: operands | equation | target */
  .focus {
    display: grid;
    grid-template-columns: auto 1fr auto;
    column-gap: var(--space-xl);
    align-items: center;
    padding: 4px 0;
  }
  @media (max-width: 760px) { .focus { grid-template-columns: 1fr; row-gap: 14px; justify-items: center; } }
  .operands { display: flex; align-items: center; gap: 10px; }
  .odot { color: var(--ink-dim); }
  .g { display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .g svg { border-radius: 2px; overflow: visible; }
  .g rect { transition: opacity 140ms ease; }
  .g rect.lit { stroke: var(--accent); stroke-width: 2.5; }
  .g rect.dim { opacity: 0.32; }
  /* no `fill` here — the per-cell adaptive fill attribute must win */
  .ct { font: 600 11px ui-monospace, Menlo, monospace; text-anchor: middle; dominant-baseline: central; pointer-events: none; }

  .eq { display: flex; flex-direction: column; gap: 3px; font-family: ui-monospace, Menlo, monospace; min-width: 300px; max-width: 420px; }
  .eq-title { font-style: italic; color: var(--accent); font-size: 0.9rem; }
  .eq-line { font-size: 0.82rem; color: var(--ink-dim); line-height: 1.5; }
  .eq-line.inter { color: var(--ink); flex-wrap: wrap; }
  .term { color: var(--ink-dim); transition: color 120ms ease, opacity 120ms ease; white-space: nowrap; }
  .term.done { color: var(--ink); }
  .term.hot { color: var(--accent-text); font-weight: 600; }
  .term.future { opacity: 0.3; }
  .eq-line.result { color: var(--ink-faint); font-weight: 600; }
  .eq-line.result.on { color: var(--accent); }

  .target { display: flex; flex-direction: column; align-items: flex-end; gap: 2px; }
  .t-lab { font-size: 0.68rem; color: var(--ink-faint); }
  .t-cell { font-size: 0.78rem; color: var(--ink-dim); }
  .t-val { font-size: 1.05rem; font-weight: 700; color: var(--accent); }
  .t-next { font-size: 0.66rem; color: var(--ink-faint); }
  @media (max-width: 760px) { .target { align-items: center; } }

</style>
