<script lang="ts">
  /**
   * PoolStep — 04. Global average pool: each of the 64 channels is a tiny 2×2
   * grid; average it into one number. Position is thrown away entirely — what's
   * left is 64 numbers that say *what* is in the sound, not *where*.
   */
  import { onDestroy, onMount } from 'svelte';
  import type { ClapBundle, Clip } from '../data/clapData';
  import { channel } from '../data/clapData';
  import { sequential } from './colormap';
  import KaTeX from '../components/Math.svelte';

  type Props = { bundle: ClapBundle; clip: Clip; n: number };
  let { bundle, clip, n }: Props = $props();

  const s3 = $derived(clip.stages.stage3);          // [2,2,64]
  const C = $derived(s3.shape[2]);                   // 64
  const latent = $derived(clip.latent.data);         // [64]
  const COLS = 8;                                    // tile/cell grid columns
  const ROWS = $derived(Math.ceil(C / COLS));

  const sLo = $derived(Math.min(...s3.data)), sHi = $derived(Math.max(...s3.data));
  const lLo = $derived(Math.min(...latent));
  const lHi = $derived(Math.max(...latent));
  function tile(c: number): number[] { return channel(s3, c); }  // 4 vals

  // ── sweep the channels, filling the pooled vector ───────────────────────────
  let cur = $state(0);
  let playing = $state(true);
  let visible = $state(true);
  let root: HTMLElement;
  let raf = 0, acc = 0, lastT = 0;
  const STEP = 90, HOLD = 1100;
  function loop(t: number) {
    raf = requestAnimationFrame(loop);
    if (!playing || !visible) { lastT = t; return; }
    if (!lastT) lastT = t;
    acc += t - lastT; lastT = t;
    const need = cur >= C - 1 ? HOLD : STEP;
    if (acc >= need) { acc = 0; cur = cur >= C - 1 ? 0 : cur + 1; }
  }
  $effect(() => { void clip; cur = 0; acc = 0; });
  onMount(() => {
    const io = new IntersectionObserver((e) => (visible = e[0].isIntersecting), { threshold: 0 });
    if (root) io.observe(root);
    raf = requestAnimationFrame(loop);
    return () => io.disconnect();
  });
  onDestroy(() => cancelAnimationFrame(raf));

  const fmt = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : '—');
  const TS = 9;   // tile cell px (2×2 → 18)
  const TG = 7;   // gap between tiles
  const VC = 17;  // pooled cell px
  const tex = String.raw`\htmlClass{part-z}{z_c} = \operatorname*{mean}_{i,j} \; X[c,i,j]`;
</script>

<section class="step" bind:this={root}>
  <header>
    <span class="eyebrow mono">{String(n).padStart(2, '0')} · Pool</span>
    <span class="dims mono">2×2×{C} → {C}</span>
  </header>

  <p class="explain">
    Stage 3 is {C} channels, each a tiny 2×2 grid. <em>Average</em> each grid into a
    single number and the spatial layout disappears. These {C} numbers say
    <em>what</em> the clip contains, not <em>where</em> in time it happened.
  </p>

  <div class="formula"><KaTeX {tex} displayMode /></div>

  <div class="row">
    <figure>
      <figcaption class="mono">stage 3 · {C} channels of 2×2</figcaption>
      <svg viewBox={`0 0 ${COLS * (2 * TS + TG)} ${ROWS * (2 * TS + TG)}`}
        width={COLS * (2 * TS + TG)} height={ROWS * (2 * TS + TG)}>
        {#each Array(C) as _, c (c)}
          {@const tx = (c % COLS) * (2 * TS + TG)}
          {@const ty = Math.floor(c / COLS) * (2 * TS + TG)}
          {#each tile(c) as v, k (k)}
            <rect x={tx + (k % 2) * TS} y={ty + Math.floor(k / 2) * TS} width={TS} height={TS} fill={sequential(v, sLo, sHi)} />
          {/each}
          {#if c === cur}<rect class="hi" x={tx} y={ty} width={2 * TS} height={2 * TS} />{/if}
        {/each}
      </svg>
    </figure>

    <div class="op mono">avg →</div>

    <figure>
      <figcaption class="mono z">pooled z · {C}</figcaption>
      <svg viewBox={`0 0 ${COLS * VC} ${ROWS * VC}`} width={COLS * VC} height={ROWS * VC}>
        {#each latent as v, c (c)}
          {@const x = (c % COLS) * VC}
          {@const y = Math.floor(c / COLS) * VC}
          <rect {x} {y} width={VC} height={VC} fill={c <= cur ? sequential(v, lLo, lHi) : 'var(--panel-2)'} />
          {#if c === cur}<rect class="hi" {x} {y} width={VC} height={VC} />{/if}
        {/each}
      </svg>
    </figure>

    <div class="op mono">=</div>

    <div class="calc mono">
      <span class="lab z">z[{cur}]</span>
      <span class="mean">mean(<span class="nums">{tile(cur).map((v) => fmt(v)).join(', ')}</span>)</span>
      <span class="res">= {fmt(latent[cur])}</span>
    </div>
  </div>
</section>

<style>
  .step { position: relative; display: flex; flex-direction: column; gap: 14px; padding: var(--space-xl) 0; border-top: 1px solid var(--line); }
  header { display: flex; align-items: baseline; gap: 14px; }
  .eyebrow { color: var(--ink-faint); font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .dims { color: var(--ink-faint); font-size: 0.72rem; margin-left: auto; }
  .explain { margin: 0; color: var(--ink-dim); font-size: 0.95rem; line-height: 1.6; max-width: 64ch; }
  .explain em { color: var(--ink); font-style: italic; }
  .formula { color: var(--ink); font-size: 0.95rem; }
  .formula :global(.part-z), .z { color: var(--data); }

  .row { display: flex; align-items: center; justify-content: center; gap: var(--space-md); flex-wrap: wrap; }
  figure { margin: 0; display: flex; flex-direction: column; align-items: center; gap: 6px; }
  figcaption { font-size: 0.7rem; color: var(--ink-dim); }
  svg { overflow: visible; image-rendering: pixelated; }
  .hi { fill: none; stroke: var(--accent); stroke-width: 1.6; vector-effect: non-scaling-stroke; transition: x 80ms linear, y 80ms linear; }
  .op { color: var(--ink-faint); font-size: 0.8rem; }

  .calc { display: flex; flex-direction: column; gap: 4px; font-size: 0.82rem; width: 200px; }
  .calc .lab { font-style: italic; }
  .mean { color: var(--ink-dim); white-space: nowrap; }
  .nums { color: var(--ink); }
  .res { color: var(--accent); font-weight: 600; }
</style>
