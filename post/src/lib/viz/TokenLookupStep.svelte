<script lang="ts">
  /**
   * 06 · Tokenise & look up. A word is just an index, and its "meaning" is one
   * column of a learned table. The table is drawn as VOCAB vertical columns (64
   * tall); each word in the label lights up its column and pulls it out as a vector.
   * The pulled column is the SAME colour and SAME size as its source column — one
   * magma scale and one geometry across the whole text side (see textViz.ts).
   */
  import { onDestroy, onMount } from 'svelte';
  import type { ClapBundle, TextItem } from '../data/clapData';
  import { sequential } from './colormap';
  import { COL_W, CELL_H, COL_GAP, featureScale } from './textViz';

  type Props = { bundle: ClapBundle; item: TextItem; n: number };
  let { bundle, item, n }: Props = $props();

  const table = $derived(bundle.weights.embTable);     // [vocab, 64]
  const VOCAB = $derived(table.shape[0]);
  const D = 64;
  const vocab = $derived(
    Object.entries(bundle.meta.vocab).sort((a, b) => (a[1] as number) - (b[1] as number))
  );
  const fs = $derived(featureScale(bundle));           // one fixed magma scale

  const words = $derived(item.tokens);
  const ids = $derived(item.ids);
  const idSet = $derived(new Set(ids));
  const raw = $derived(item.rawEmbed.data);            // [T,64]
  const T = $derived(words.length);

  const tableCol = (wd: number) => Array.from({ length: D }, (_, d) => table.data[wd * D + d]);
  const rawCol = (r: number) => Array.from({ length: D }, (_, d) => raw[r * D + d]);

  // ── sweep the words, each one lighting up + pulling its column ───────────────
  let wi = $state(0);
  let visible = $state(true);
  let root: HTMLElement;
  let raf = 0, acc = 0, lastT = 0;
  const DWELL = 1300;
  function loop(t: number) {
    raf = requestAnimationFrame(loop);
    if (!visible) { lastT = t; return; }
    if (!lastT) lastT = t;
    acc += t - lastT; lastT = t;
    if (acc >= DWELL) { acc = 0; wi = (wi + 1) % Math.max(1, T); }
  }
  $effect(() => { void item; wi = 0; acc = 0; });
  onMount(() => {
    const io = new IntersectionObserver((e) => (visible = e[0].isIntersecting), { threshold: 0 });
    if (root) io.observe(root);
    raf = requestAnimationFrame(loop);
    return () => io.disconnect();
  });
  onDestroy(() => cancelAnimationFrame(raf));

  const curId = $derived(ids[wi] ?? 0);
</script>

{#snippet column(vals: number[])}
  <svg class="bar" viewBox={`0 0 ${COL_W} ${vals.length * CELL_H}`} width={COL_W} height={vals.length * CELL_H}>
    {#each vals as v, i (i)}
      <rect x={0} y={i * CELL_H} width={COL_W} height={CELL_H} fill={sequential(v, fs.lo, fs.hi)} />
    {/each}
  </svg>
{/snippet}

<section class="step" bind:this={root}>
  <header>
    <span class="eyebrow mono">{String(n).padStart(2, '0')} · Tokenise &amp; look up</span>
    <span class="dims mono">words → vectors</span>
  </header>

  <p class="explain">
    The text side has to reach the <em>same</em> 512-d space as the sound, and its
    first move is the one worth slowing down for. A word isn't processed as letters
    — it's turned into an <em>index</em>, and that index points at one column of a
    learned table. The column <em>is</em> the word's meaning, 64 numbers the model
    tuned during training. Each word in the label pulls its column out — same column,
    same colours.
  </p>

  <!-- words → ids -->
  <div class="tokens">
    {#each words as w, i (i)}
      <div class="tok" class:on={i === wi}>
        <span class="word">{w}</span>
        <span class="id mono">#{ids[i]}</span>
      </div>
    {/each}
  </div>

  <div class="flow">
    <!-- the whole table as vocab columns -->
    <figure>
      <figcaption class="mono d">embedding table · {VOCAB} words × {D}</figcaption>
      <div class="cols" style={`gap:${COL_GAP}px`}>
        {#each vocab as [w, id] (id)}
          <div class="col" class:on={id === curId} class:inlabel={idSet.has(id as number)}>
            {@render column(tableCol(id as number))}
            <span class="vlabel mono">{w}</span>
          </div>
        {/each}
      </div>
    </figure>

    <div class="bridge mono">
      <span class="big">id&nbsp;#<span class="num">{curId}</span></span>
      <span>picks a column</span>
      <span class="ar">→</span>
    </div>

    <!-- the pulled-out word vectors: identical columns -->
    <figure>
      <figcaption class="mono d">word vectors · {T} × {D}</figcaption>
      <div class="cols" style={`gap:${COL_GAP}px`}>
        {#each words as w, r (r)}
          <div class="col" class:on={r === wi}>
            {@render column(rawCol(r))}
            <span class="vlabel mono">{w}</span>
          </div>
        {/each}
      </div>
    </figure>
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

  .tokens { display: flex; gap: 10px; flex-wrap: wrap; }
  .tok { display: flex; flex-direction: column; align-items: center; gap: 2px; padding: 7px 14px;
    border: 1px solid var(--line-strong); border-radius: 7px; background: var(--panel); transition: all 160ms; }
  .tok.on { border-color: var(--accent); background: var(--bg); box-shadow: 0 0 0 1px var(--accent); }
  .tok .word { font-size: 0.98rem; color: var(--ink); }
  .tok .id { font-size: 0.7rem; color: var(--accent); }

  .flow { display: flex; align-items: flex-start; justify-content: center; gap: var(--space-xl); flex-wrap: wrap; margin-top: 6px; }
  figure { margin: 0; display: flex; flex-direction: column; align-items: center; gap: 12px; }
  figcaption { font-size: 0.72rem; color: var(--ink-dim); }

  .cols { display: flex; align-items: flex-start; }
  .col { display: flex; flex-direction: column; align-items: center; gap: 6px; opacity: 0.45; transition: opacity 200ms; }
  .col.inlabel { opacity: 0.95; }          /* words that appear in this caption */
  .col.on { opacity: 1; }
  .bar { image-rendering: pixelated; box-shadow: 0 0 0 1px var(--line); border-radius: 2px; transition: box-shadow 160ms; }
  .col.on .bar { box-shadow: 0 0 0 2px var(--accent); }
  .vlabel { font-size: 0.66rem; color: var(--ink-faint); writing-mode: vertical-rl; text-orientation: mixed;
    max-height: 64px; transition: color 160ms; }
  .col.on .vlabel { color: var(--accent); font-weight: 600; }

  .bridge { display: flex; flex-direction: column; align-items: center; gap: 4px; align-self: center;
    color: var(--ink-faint); font-size: 0.74rem; }
  .bridge .big { color: var(--accent); font-weight: 600; font-size: 0.9rem; }
  .bridge .num { display: inline-block; min-width: 1.1ch; text-align: left; font-variant-numeric: tabular-nums; }
  .bridge .ar { font-size: 1.1rem; margin-top: 2px; }
</style>
