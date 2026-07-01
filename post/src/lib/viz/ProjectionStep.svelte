<script lang="ts">
  /**
   * ProjectionStep — 05, the finale. A matrix multiply lifts the 64 pooled
   * numbers into the 512-dim shared space: each of the 512 outputs is one weight
   * row dotted with z. We sweep the rows so you watch 64 numbers fan out into
   * 512; then ReLU, a second 512→512, and unit-normalisation give the embedding
   * — the same kind of object the text side produces, so the two can be compared.
   */
  import { onDestroy, onMount } from 'svelte';
  import type { ClapBundle, Clip } from '../data/clapData';
  import { sequential, diverging } from './colormap';
  import { EMB_CELL, EMB_COLS, embedScale } from './textViz';
  import KaTeX from '../components/Math.svelte';

  type Props = { bundle: ClapBundle; clip: Clip; n: number };
  let { bundle, clip, n }: Props = $props();

  const z = $derived(clip.latent.data);             // 64
  const W1 = $derived(bundle.weights.proj1);        // [512,64]
  const b1 = $derived(bundle.weights.proj1Bias);
  const embed = $derived(clip.audioEmbed);          // 512 (final)
  const OUT = 512;

  const zLo = $derived(Math.min(...z)), zHi = $derived(Math.max(...z));
  const es = $derived(embedScale(bundle));   // shared 512 scale (matches text 08)
  // hidden pre-activation built by the first matmul (real)
  const h = $derived.by(() => {
    const out = new Array(OUT);
    for (let j = 0; j < OUT; j++) {
      let s = b1.data[j];
      for (let k = 0; k < 64; k++) s += z[k] * W1.data[j * 64 + k];
      out[j] = Math.max(0, s);
    }
    return out;
  });
  const hAbs = $derived(Math.max(1e-6, ...h.map(Math.abs)));
  // outputs that actually fire (ReLU > 0) — we only ever *explain* these, so the
  // worked example never lands on a dead 0.00. The full 512 are still shown.
  const live = $derived(h.map((v, j) => (v > 1e-4 ? j : -1)).filter((j) => j >= 0));
  const row = $derived.by(() => {
    const r: number[] = [];
    for (let k = 0; k < 64; k++) r.push(W1.data[jcur * 64 + k]);
    return r;
  });
  const rAbs = $derived(Math.max(1e-6, ...row.map(Math.abs)));

  // ── dwell on one firing output, accumulate its 64-term dot product, advance ──
  let ptr = $state(0);    // index into `live`
  let kk = $state(0);     // how many of its 64 terms are summed in (0…64)
  const jcur = $derived(live.length ? live[ptr % live.length] : 0);  // actual output 0…511
  let playing = $state(true);
  let visible = $state(true);
  let root: HTMLElement;
  let raf = 0, acc = 0, lastT = 0;
  const KSTEP = 24;       // ms per term as the sum builds
  const HOLD = 700;       // pause on the finished output before moving on
  function loop(t: number) {
    raf = requestAnimationFrame(loop);
    if (!playing || !visible) { lastT = t; return; }
    if (!lastT) lastT = t;
    acc += t - lastT; lastT = t;
    if (kk < 64) {
      if (acc >= KSTEP) { acc = 0; kk++; }
    } else if (acc >= HOLD) {
      acc = 0; kk = 0; ptr = live.length ? (ptr + 1) % live.length : 0;
    }
  }
  // running partial sum b_j + Σ_{k<kk} z_k·W_{j,k}; full pre-ReLU sum for the result line
  const partial = $derived.by(() => {
    let s = b1.data[jcur];
    for (let k = 0; k < Math.min(kk, 64); k++) s += z[k] * W1.data[jcur * 64 + k];
    return s;
  });
  const fullSum = $derived.by(() => {
    let s = b1.data[jcur];
    for (let k = 0; k < 64; k++) s += z[k] * W1.data[jcur * 64 + k];
    return s;
  });
  $effect(() => { void clip; ptr = 0; kk = 0; acc = 0; });
  onMount(() => {
    const io = new IntersectionObserver((e) => (visible = e[0].isIntersecting), { threshold: 0 });
    if (root) io.observe(root);
    raf = requestAnimationFrame(loop);
    return () => io.disconnect();
  });
  onDestroy(() => cancelAnimationFrame(raf));

  const fmt = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : '—');
  const ZC = 16;          // z cell
  const EC = EMB_CELL, ECOLS = EMB_COLS; // shared 512 cell + cols (audio + text match)
  const RC = 7;           // weight-row cell
  const tex = String.raw`\htmlClass{part-h}{h_j} = b_j + \textstyle\sum_k \htmlClass{part-w}{W_{j,k}}\,\htmlClass{part-z}{z_k} \qquad E = \tfrac{\,\operatorname{ReLU}(h)\,W_2\,}{\lVert\cdot\rVert}`;
</script>

<section class="step" bind:this={root}>
  <header>
    <span class="eyebrow mono">{String(n).padStart(2, '0')} · Project to 512</span>
    <span class="dims mono">64 → 512 · the audio embedding</span>
  </header>

  <p class="explain">
    Finally, a matrix multiply lifts the {64} pooled numbers into the shared
    512-dimensional space — each output is one row of weights dotted with
    <span class="z">z</span>. Watch 64 numbers fan out into 512; a ReLU, a second
    512→512, and a unit-normalise turn that into the <em>embedding</em>: this
    sound, as a single point in the space CLAP shares with words.
  </p>

  <div class="formula"><KaTeX {tex} displayMode /></div>

  <div class="row">
    <figure>
      <figcaption class="mono z">z · 64</figcaption>
      <svg viewBox={`0 0 ${8 * ZC} ${8 * ZC}`} width={8 * ZC} height={8 * ZC}>
        {#each z as v, k (k)}
          <rect x={(k % 8) * ZC} y={Math.floor(k / 8) * ZC} width={ZC} height={ZC} fill={sequential(v, zLo, zHi)} />
          {#if kk < 64 && k === kk}<rect class="hi" x={(k % 8) * ZC} y={Math.floor(k / 8) * ZC} width={ZC} height={ZC} />{/if}
        {/each}
      </svg>
    </figure>

    <div class="mid">
      <span class="dotlab mono">· W<tspan>1</tspan>[{jcur}] =</span>
      <svg class="wrow" viewBox={`0 0 ${64 * RC} ${RC}`} preserveAspectRatio="none">
        {#each row as v, k (k)}
          <rect x={k * RC} y="0" width={RC} height={RC} fill={diverging(v, rAbs)} />
          {#if kk < 64 && k === kk}<rect class="hi" x={k * RC} y="0" width={RC} height={RC} />{/if}
        {/each}
      </svg>
      <!-- the dot product, worked one term at a time like step 02 -->
      <div class="eq mono">
        <div class="eq-def"><span class="z">h[{jcur}]</span> = b<tspan>j</tspan> + Σ<tspan>k</tspan> <span class="z">z<tspan>k</tspan></span>·<span class="w">W<tspan>j,k</tspan></span></div>
        <div class="eq-run">
          = <span class="z">{fmt(partial)}</span>{#if kk < 64}<span class="term">{' + '}<span class="z">{fmt(z[kk])}</span>{' · '}<span class="w">{fmt(row[kk])}</span> <span class="kx">(k={kk})</span></span>{/if}
        </div>
        <div class="eq-res" class:on={kk >= 64}>
          = ReLU({fmt(fullSum)}) = <span class="z hot">{fmt(h[jcur])}</span>
        </div>
      </div>
    </div>

    <figure>
      <figcaption class="mono">hidden h · 512</figcaption>
      <svg viewBox={`0 0 ${ECOLS * EC} ${Math.ceil(OUT / ECOLS) * EC}`} width={ECOLS * EC} height={Math.ceil(OUT / ECOLS) * EC}>
        {#each h as v, j (j)}
          <rect x={(j % ECOLS) * EC} y={Math.floor(j / ECOLS) * EC} width={EC} height={EC}
            fill={sequential(v, 0, hAbs)} opacity={j === jcur ? 1 : 0.42} />
          {#if j === jcur}<rect class="hi" x={(j % ECOLS) * EC} y={Math.floor(j / ECOLS) * EC} width={EC} height={EC} />{/if}
        {/each}
      </svg>
    </figure>
  </div>

  <div class="finale">
    <span class="arrow mono">ReLU · W₂ · unit-normalise →</span>
    <figure>
      <figcaption class="mono final">the audio embedding · 512</figcaption>
      <svg viewBox={`0 0 ${ECOLS * EC} ${Math.ceil(OUT / ECOLS) * EC}`} width={ECOLS * EC} height={Math.ceil(OUT / ECOLS) * EC}>
        {#each embed as v, j (j)}
          <rect x={(j % ECOLS) * EC} y={Math.floor(j / ECOLS) * EC} width={EC} height={EC} fill={sequential(v, es.lo, es.hi)} />
        {/each}
      </svg>
    </figure>
  </div>
</section>

<style>
  .step { position: relative; display: flex; flex-direction: column; gap: 14px; padding: var(--space-xl) 0; border-top: 1px solid var(--line); }
  header { display: flex; align-items: baseline; gap: 14px; }
  .eyebrow { color: var(--ink-faint); font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .dims { color: var(--ink-faint); font-size: 0.72rem; margin-left: auto; }
  .explain { margin: 0; color: var(--ink-dim); font-size: 0.95rem; line-height: 1.6; max-width: 64ch; }
  .explain em { color: var(--ink); font-style: italic; }
  .z { color: var(--data); }
  .formula { color: var(--ink); font-size: 0.92rem; }
  .formula :global(.part-h), .formula :global(.part-z) { color: var(--data); }
  .formula :global(.part-w) { color: var(--weight); }

  .row { display: flex; align-items: center; justify-content: center; gap: var(--space-md); flex-wrap: wrap; }
  figure { margin: 0; display: flex; flex-direction: column; align-items: center; gap: 6px; }
  figcaption { font-size: 0.7rem; color: var(--ink-dim); }
  .final { color: var(--data); font-weight: 600; }
  svg { overflow: visible; image-rendering: pixelated; box-shadow: 0 0 0 1px var(--line); border-radius: 2px; }
  .mid { display: flex; flex-direction: column; align-items: center; gap: 6px; max-width: 100%; min-width: 0; }
  .dotlab { font-size: 0.74rem; color: var(--ink-dim); }
  .dotlab tspan, .eq tspan { font-size: 0.6em; vertical-align: sub; }
  .wrow { box-shadow: 0 0 0 1px var(--line); width: 100%; max-width: 448px; height: 7px; }
  .hi { fill: none; stroke: var(--accent); stroke-width: 1.6; vector-effect: non-scaling-stroke; }

  /* the worked dot product — built one term at a time, like step 02 */
  .eq { display: flex; flex-direction: column; align-items: flex-start; gap: 3px;
        font-size: 0.78rem; min-width: 248px; line-height: 1.5; }
  .eq .z { color: var(--data); }
  .eq .w { color: var(--weight); }
  .eq-def { color: var(--ink-dim); }
  .eq-run { color: var(--ink); }
  .eq-run .term { color: var(--ink-dim); }
  .eq-run .kx { color: var(--ink-faint); font-size: 0.85em; }
  .eq-res { color: var(--ink-faint); opacity: 0.35; transition: opacity 140ms; }
  .eq-res.on { opacity: 1; color: var(--ink-dim); }
  .eq-res .hot { font-weight: 700; }

  .finale { display: flex; align-items: center; justify-content: center; gap: var(--space-md); flex-wrap: wrap; margin-top: 4px; }
  .arrow { color: var(--ink-faint); font-size: 0.74rem; white-space: nowrap; }
</style>
