<script lang="ts">
  /**
   * 10 · The contrastive matrix. Zoom out from one clip to all of them: every
   * sound (rows) dotted with every caption (columns). This grid IS the training
   * objective — push the diagonal (each sound with its own caption) up, everything
   * else down. Values are the real captured cosines (similarity.json).
   */
  import type { ClapBundle, Clip } from '../data/clapData';

  type Props = { bundle: ClapBundle; clip: Clip; n: number };
  let { bundle, clip, n }: Props = $props();

  const sim = $derived(bundle.similarity);
  const N = $derived(sim.slugs.length);
  const cos = $derived(sim.cosine);            // [audio][text]
  const captions = $derived(sim.captions);
  const activeRow = $derived(sim.slugs.indexOf(clip.slug));

  // amber intensity for a cosine: negatives ~transparent, +1 → full --data
  const alpha = (c: number) => Math.max(0.04, Math.min(1, (c + 0.15) / 1.05));
  const ink = (c: number) => (alpha(c) > 0.55 ? '#fbf3df' : '#5a4a38');
  const fmt = (x: number) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(2);
</script>

<section class="step">
  <header>
    <span class="eyebrow mono">{String(n).padStart(2, '0')} · The contrastive matrix</span>
    <span class="dims mono">{N} sounds × {N} captions</span>
  </header>

  <p class="explain">
    The match runs one clip against the labels; step back and run <em>every</em>
    sound against <em>every</em> caption and you get this grid — each cell is one
    cosine. This is the whole training objective in a picture: pull the
    <em>diagonal</em> up (each sound with its own words) and push everything off it
    down. A perfect model is a bright diagonal on a dark field.
  </p>

  <div class="matrix" style={`--n:${N}`}>
    <div class="corner mono">cos(a, t)</div>
    {#each captions as cap, j (j)}
      <div class="colhead mono" style={`grid-column:${j + 2}`}>{cap}</div>
    {/each}

    {#each sim.slugs as slug, i (slug)}
      <div class="rowhead mono" class:on={i === activeRow} style={`grid-row:${i + 2}`}>{captions[i]}</div>
      {#each cos[i] as c, j (j)}
        <div
          class="cell mono"
          class:diag={i === j}
          class:activerow={i === activeRow}
          style={`grid-column:${j + 2}; grid-row:${i + 2}; background:rgba(184,85,31,${alpha(c)}); color:${ink(c)}`}
        >{fmt(c)}</div>
      {/each}
    {/each}
  </div>

  <p class="foot">
    Our diagonal reads {cos.map((r, i) => fmt(r[i])).join(', ')} — bright — while the
    off-diagonal cells sit near zero or negative. That gap is what a contrastive loss
    buys, and it's why a sound can be labelled with words it was never trained
    against.
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

  .matrix {
    display: grid;
    grid-template-columns: max-content repeat(var(--n), 1fr);
    grid-auto-rows: 58px;
    gap: 4px;
    align-items: stretch;
    max-width: 640px;
    margin: 4px auto 0;
    width: 100%;
  }
  .corner { grid-column: 1; grid-row: 1; align-self: end; justify-self: end;
    font-size: 0.68rem; color: var(--ink-faint); padding: 0 8px 6px 0; }
  .colhead { grid-row: 1; align-self: end; text-align: center; padding-bottom: 6px;
    font-size: 0.72rem; color: var(--ink-dim); line-height: 1.25; }
  .rowhead { grid-column: 1; align-self: center; justify-self: end; text-align: right;
    padding-right: 10px; font-size: 0.72rem; color: var(--ink-dim); max-width: 128px; line-height: 1.2; }
  .rowhead.on { color: var(--accent); font-weight: 600; }

  .cell { display: flex; align-items: center; justify-content: center;
    border-radius: 4px; font-size: 0.82rem; font-variant-numeric: tabular-nums;
    box-shadow: inset 0 0 0 1px var(--line); transition: transform 120ms; }
  .cell.diag { box-shadow: inset 0 0 0 2px var(--accent); }
  .cell.activerow { outline: 2px solid var(--accent); outline-offset: 1px; }

  @media (max-width: 560px) {
    .matrix { grid-auto-rows: 46px; }
    .colhead, .rowhead { font-size: 0.62rem; }
    .rowhead { max-width: 84px; }
    .cell { font-size: 0.72rem; }
  }
</style>
