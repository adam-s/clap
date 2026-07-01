<script lang="ts">
  /**
   * 08 · Project → 512. The text side ends with the same move as the audio side:
   * 64 → 512, ReLU, 512 → 512, L2-normalise. It's a *separate* copy of those
   * weights (audio and text have their own heads), trained at the same time so the
   * two outputs land in one comparable space. Result drawn as a 512 block, not a
   * thin strip, so it reads next to the audio embedding from step 05.
   */
  import type { ClapBundle, TextItem } from '../data/clapData';
  import { sequential } from './colormap';
  import { COL_W, CELL_H, EMB_CELL, EMB_COLS, featureScale, embedScale } from './textViz';
  import KaTeX from '../components/Math.svelte';

  type Props = { bundle: ClapBundle; item: TextItem; n: number };
  let { bundle, item, n }: Props = $props();

  const D = 64;
  const latent = $derived(item.latent);     // [64]
  const embed = $derived(item.textEmbed);   // [512]
  const fs = $derived(featureScale(bundle)); // shared feature scale (matches 06–07)
  const es = $derived(embedScale(bundle));   // shared 512 scale (matches audio 05)

  const tex = String.raw`E_{\text{text}} = \frac{\operatorname{ReLU}(z\,W_1)\,W_2}{\lVert\cdot\rVert}`;
</script>

<section class="step">
  <header>
    <span class="eyebrow mono">{String(n).padStart(2, '0')} · Project to 512</span>
    <span class="dims mono">64 → 512 · the text embedding</span>
  </header>

  <p class="explain">
    Same finish as the audio side: 64 → 512, ReLU, 512 → 512, normalise. The one
    difference matters. The text encoder keeps its <em>own</em> copy of these weights;
    the two heads never share parameters. Training them together is the only thing
    that lines a caption up with its sound.
  </p>

  <div class="formula"><KaTeX {tex} displayMode /></div>

  <div class="flow">
    <figure>
      <figcaption class="mono d">pooled z · {D}</figcaption>
      <svg class="bar" viewBox={`0 0 ${COL_W} ${D * CELL_H}`} width={COL_W} height={D * CELL_H}>
        {#each latent as v, i (i)}
          <rect x={0} y={i * CELL_H} width={COL_W} height={CELL_H} fill={sequential(v, fs.lo, fs.hi)} />
        {/each}
      </svg>
    </figure>

    <div class="bridge mono">
      <span>ReLU · W₂</span>
      <span>normalise</span>
      <span class="ar">→</span>
    </div>

    <figure>
      <figcaption class="mono d final">the text embedding · 512</figcaption>
      <svg class="bar" viewBox={`0 0 ${EMB_COLS * EMB_CELL} ${Math.ceil(512 / EMB_COLS) * EMB_CELL}`}
        width={EMB_COLS * EMB_CELL} height={Math.ceil(512 / EMB_COLS) * EMB_CELL}>
        {#each embed as v, j (j)}
          <rect x={(j % EMB_COLS) * EMB_CELL} y={Math.floor(j / EMB_COLS) * EMB_CELL} width={EMB_CELL} height={EMB_CELL} fill={sequential(v, es.lo, es.hi)} />
        {/each}
      </svg>
    </figure>
  </div>

  <p class="foot">
    That's it — "{item.caption}" now lives beside the sound in one space. All that's
    left is to measure the distance.
  </p>
</section>

<style>
  .step { position: relative; display: flex; flex-direction: column; gap: 16px; padding: var(--space-2xl) 0; border-top: 1px solid var(--line); }
  header { display: flex; align-items: baseline; gap: 14px; }
  .eyebrow { color: var(--ink-faint); font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .dims { color: var(--ink-faint); font-size: 0.72rem; margin-left: auto; }
  .explain, .foot { margin: 0; color: var(--ink-dim); font-size: 0.95rem; line-height: 1.6; max-width: 64ch; }
  .explain em { color: var(--ink); font-style: italic; }
  .foot { font-size: 0.9rem; color: var(--ink-faint); }
  .d { color: var(--data); }
  .final { color: var(--data); font-weight: 600; }
  .formula { color: var(--ink); font-size: 0.92rem; }
  .formula :global(.katex) { color: var(--ink-dim); }

  .flow { display: flex; align-items: center; justify-content: center; gap: var(--space-lg); flex-wrap: wrap; }
  figure { margin: 0; display: flex; flex-direction: column; align-items: center; gap: 12px; }
  figcaption { font-size: 0.72rem; color: var(--ink-dim); }
  .bar { image-rendering: pixelated; box-shadow: 0 0 0 1px var(--line); border-radius: 2px; }
  .bridge { display: flex; flex-direction: column; align-items: center; gap: 2px; color: var(--ink-faint); font-size: 0.72rem; }
  .bridge .ar { font-size: 1.1rem; }
</style>
