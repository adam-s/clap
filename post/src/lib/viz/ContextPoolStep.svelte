<script lang="ts">
  /**
   * 07 · Mix & pool → 64. The word columns go through a tiny Transformer (each
   * column updated by reading the others — context), then we average the columns
   * into one 64-vector. Same magma scale and column geometry as step 06, so a
   * value keeps its colour; we sweep a dimension and show the mean.
   */
  import { onDestroy, onMount } from 'svelte';
  import type { ClapBundle, TextItem } from '../data/clapData';
  import { sequential } from './colormap';
  import { COL_W, CELL_H, COL_GAP, featureScale } from './textViz';

  type Props = { bundle: ClapBundle; item: TextItem; n: number };
  let { bundle, item, n }: Props = $props();

  const D = 64;
  const words = $derived(item.tokens);
  const T = $derived(words.length);
  const mixed = $derived(item.tokenEmbed.data);   // [T,64] after context mixing
  const latent = $derived(item.latent);           // [64] pooled
  const fs = $derived(featureScale(bundle));       // shared magma scale
  const mixedCol = (r: number) => Array.from({ length: D }, (_, d) => mixed[r * D + d]);

  // ── sweep a dimension, showing its mean across the word columns ──────────────
  let d = $state(0);
  let visible = $state(true);
  let root: HTMLElement;
  let raf = 0, acc = 0, lastT = 0;
  const STEP = 150;
  function loop(t: number) {
    raf = requestAnimationFrame(loop);
    if (!visible) { lastT = t; return; }
    if (!lastT) lastT = t;
    acc += t - lastT; lastT = t;
    if (acc >= STEP) { acc = 0; d = (d + 1) % D; }
  }
  $effect(() => { void item; d = 0; acc = 0; });
  onMount(() => {
    const io = new IntersectionObserver((e) => (visible = e[0].isIntersecting), { threshold: 0 });
    if (root) io.observe(root);
    raf = requestAnimationFrame(loop);
    return () => io.disconnect();
  });
  onDestroy(() => cancelAnimationFrame(raf));

  const fmt = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : '—');
</script>

{#snippet column(vals: number[])}
  <svg class="bar" viewBox={`0 0 ${COL_W} ${vals.length * CELL_H}`} width={COL_W} height={vals.length * CELL_H}>
    {#each vals as v, i (i)}
      <rect x={0} y={i * CELL_H} width={COL_W} height={CELL_H} fill={sequential(v, fs.lo, fs.hi)} />
    {/each}
    <rect class="band" x={0} y={d * CELL_H} width={COL_W} height={CELL_H} />
  </svg>
{/snippet}

<section class="step" bind:this={root}>
  <header>
    <span class="eyebrow mono">{String(n).padStart(2, '0')} · Mix &amp; pool</span>
    <span class="dims mono">{T}×64 → 64</span>
  </header>

  <p class="explain">
    A handful of word-columns isn't a phrase yet. A small Transformer lets every
    column <em>read the others</em>, so each one picks up context from its
    neighbours. In this small model the columns come out looking
    <em>nearly identical</em> — they started far apart, and after mixing they barely
    differ. That's why the averaging in the next step loses almost nothing: it
    collapses <em>columns that already agree</em> into 64 numbers for the whole
    phrase.
  </p>

  <div class="flow">
    <figure>
      <figcaption class="mono d">mixed word columns · {T}×{D}</figcaption>
      <div class="cols" style={`gap:${COL_GAP}px`}>
        {#each words as w, r (r)}
          <div class="col">
            {@render column(mixedCol(r))}
            <span class="vlabel mono">{w}</span>
          </div>
        {/each}
      </div>
      <span class="hint mono">↑ in this model, mixing left the columns nearly identical</span>
    </figure>

    <div class="bridge mono">
      <span>average row&nbsp;<span class="num">{d}</span></span>
      <span class="ar">→</span>
    </div>

    <figure>
      <figcaption class="mono d">pooled z · {D}</figcaption>
      <div class="col">
        {@render column(latent)}
        <span class="vlabel mono">z</span>
      </div>
    </figure>

    <div class="calc mono">
      <span class="lab d">z[<span class="num">{d}</span>]</span>
      <span class="mean">= mean(<span class="nums">{Array.from({ length: T }, (_, r) => fmt(mixed[r * D + d])).join(', ')}</span>)</span>
      <span class="res">= {fmt(latent[d])}</span>
    </div>
  </div>
</section>

<style>
  .step { position: relative; display: flex; flex-direction: column; gap: 16px; padding: var(--space-2xl) 0; border-top: 1px solid var(--line); }
  header { display: flex; align-items: baseline; gap: 14px; }
  .eyebrow { color: var(--ink-faint); font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .dims { color: var(--ink-faint); font-size: 0.72rem; margin-left: auto; }
  .explain { margin: 0; color: var(--ink-dim); font-size: 0.95rem; line-height: 1.6; max-width: 64ch; }
  .explain em { color: var(--ink); font-style: italic; }
  .d { color: var(--data); }

  .flow { display: flex; align-items: flex-start; justify-content: center; gap: var(--space-lg); flex-wrap: wrap; margin-top: 6px; }
  figure { margin: 0; display: flex; flex-direction: column; align-items: center; gap: 12px; }
  figcaption { font-size: 0.72rem; color: var(--ink-dim); }
  .cols { display: flex; align-items: flex-start; }
  .col { display: flex; flex-direction: column; align-items: center; gap: 6px; }
  .bar { image-rendering: pixelated; box-shadow: 0 0 0 1px var(--line); border-radius: 2px; overflow: visible; }
  .band { fill: none; stroke: var(--accent); stroke-width: 1.4; vector-effect: non-scaling-stroke; transition: y 120ms linear; }
  .vlabel { font-size: 0.66rem; color: var(--ink-faint); writing-mode: vertical-rl; text-orientation: mixed; max-height: 64px; }
  .hint { font-size: 0.66rem; color: var(--ink-faint); }

  .bridge { display: flex; flex-direction: column; align-items: center; gap: 2px; align-self: center;
    width: 116px; text-align: center; color: var(--ink-faint); font-size: 0.74rem; }
  .bridge .ar { font-size: 1.1rem; }
  /* fixed slot for the 0–63 index so changing digits never reflow anything */
  .num { display: inline-block; width: 2ch; text-align: right; font-variant-numeric: tabular-nums; }

  .calc { display: flex; flex-direction: column; gap: 5px; font-size: 0.82rem; width: 260px; align-self: center; }
  .calc .lab { font-style: italic; font-weight: 600; }
  .mean { color: var(--ink-dim); white-space: nowrap; }
  .nums { color: var(--ink); }
  .res { color: var(--accent); font-weight: 600; }
</style>
