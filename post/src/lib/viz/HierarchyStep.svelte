<script lang="ts">
  /**
   * HierarchyStep — 03. The convolution's mechanics were shown in 02; here the
   * SAME operation just repeats, cascading down the stack. We animate one conv's
   * full sweep at a time — its window sliding over the input while the output
   * fills — then hand off to the next. Because each stage has a different number
   * of positions (256 → 64 → 16 → 4), running them in sequence (not in lockstep)
   * is what actually reads cleanly.
   */
  import { onDestroy, onMount } from 'svelte';
  import type { ClapBundle, Clip, TensorMap } from '../data/clapData';
  import { channelMean } from '../data/clapData';
  import { sequential } from './colormap';

  type Props = { bundle: ClapBundle; clip: Clip; n: number };
  let { bundle, clip, n }: Props = $props();

  const maps = $derived([
    { key: 'mel', label: 'mel', map: clip.mel, size: bundle.meta.img, ch: 1, cell: 4 },
    { key: 'patch', label: 'patch', map: clip.stages.patch, size: 16, ch: 8, cell: 7 },
    { key: 'stage1', label: 'stage 1', map: clip.stages.stage1, size: 8, ch: 16, cell: 13 },
    { key: 'stage2', label: 'stage 2', map: clip.stages.stage2, size: 4, ch: 32, cell: 22 },
    { key: 'stage3', label: 'stage 3', map: clip.stages.stage3, size: 2, ch: 64, cell: 40 },
  ] as { key: string; label: string; map: TensorMap; size: number; ch: number; cell: number }[]);

  function meanCells(m: TensorMap): { vals: number[]; lo: number; hi: number } {
    const vals = m.shape[2] === 1 ? m.data : channelMean(m);
    let lo = Infinity, hi = -Infinity;
    for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
    return { vals, lo, hi };
  }

  // ── cascade: one conv's full sweep at a time ────────────────────────────────
  let conv = $state(0);    // 0..3, which transition (maps[conv] → maps[conv+1])
  let pos = $state(0);     // output cell being computed in this conv
  let playing = $state(true);
  let visible = $state(true);
  let root: HTMLElement;
  let raf = 0, acc = 0, lastT = 0;
  const DUR = 2600;        // wall-clock per conv (each at its own cell-rate)

  const outSize = $derived(maps[conv + 1].size);
  const oi = $derived(Math.floor(pos / outSize));
  const oj = $derived(pos % outSize);

  function loop(t: number) {
    raf = requestAnimationFrame(loop);
    if (!playing || !visible) { lastT = t; return; }
    if (!lastT) lastT = t;
    acc += t - lastT; lastT = t;
    const cells = maps[conv + 1].size ** 2;
    pos = Math.min(cells - 1, Math.floor((acc / DUR) * cells));
    if (acc >= DUR) { acc = 0; pos = 0; conv = (conv + 1) % 4; }
  }
  $effect(() => { void clip; conv = 0; pos = 0; acc = 0; });
  onMount(() => {
    const io = new IntersectionObserver((e) => (visible = e[0].isIntersecting), { threshold: 0 });
    if (root) io.observe(root);
    raf = requestAnimationFrame(loop);
    return () => io.disconnect();
  });
  onDestroy(() => cancelAnimationFrame(raf));

  // per-map render mode: done (full) · filling (0..pos) · pending (dim)
  function mode(L: number): 'done' | 'filling' | 'pending' {
    if (L <= conv) return 'done';
    if (L === conv + 1) return 'filling';
    return 'pending';
  }
</script>

<section class="step" bind:this={root}>
  <header>
    <span class="eyebrow mono">{String(n).padStart(2, '0')} · The hierarchy</span>
    <span class="dims mono">patch → stage 3 · 16→8→4→2, 8→16→32→64</span>
  </header>

  <p class="explain">
    From here the same 3×3 convolution just repeats, cascading down the stack:
    each pass sweeps the whole map, halves the resolution, and doubles the
    detectors. Watch one finish and hand the result to the next — by the end, a
    2×2 grid of 64 numbers stands for the entire clip.
  </p>

  <div class="chain">
    {#each maps as m, L (m.key)}
      {@const mc = meanCells(m.map)}
      {@const md = mode(L)}
      <figure class:pending={md === 'pending'}>
        <figcaption class="mono" class:active={L === conv || L === conv + 1}>{m.label}<span class="dim"> · {m.size}²×{m.ch}</span></figcaption>
        <svg viewBox={`0 0 ${m.size * m.cell} ${m.size * m.cell}`} width={m.size * m.cell} height={m.size * m.cell}>
          {#each mc.vals as v, k (k)}
            <rect x={(k % m.size) * m.cell} y={Math.floor(k / m.size) * m.cell} width={m.cell} height={m.cell}
              fill={md === 'filling' && k > pos ? 'var(--panel-2)' : sequential(v, mc.lo, mc.hi)} />
          {/each}
          {#if L === conv}
            <rect class="win" x={(2 * oj - 1) * m.cell} y={(2 * oi - 1) * m.cell} width={3 * m.cell} height={3 * m.cell} />
          {/if}
          {#if md === 'filling'}
            <rect class="cur" x={oj * m.cell} y={oi * m.cell} width={m.cell} height={m.cell} />
          {/if}
        </svg>
      </figure>
      {#if L < maps.length - 1}<span class="conv mono" class:active={L === conv}>3×3<br />⁄2 →</span>{/if}
    {/each}
  </div>

  <aside class="rf-note">
    a 3×3 kernel, stacked four deep, ends up seeing <em>almost the whole picture</em>
  </aside>
</section>

<style>
  .step { position: relative; display: flex; flex-direction: column; gap: 14px; padding: var(--space-xl) 0; border-top: 1px solid var(--line); }
  header { display: flex; align-items: baseline; gap: 14px; }
  .eyebrow { color: var(--ink-faint); font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .dims { color: var(--ink-faint); font-size: 0.72rem; margin-left: auto; }
  .explain { margin: 0; color: var(--ink-dim); font-size: 0.95rem; line-height: 1.6; max-width: 64ch; }

  .chain { display: flex; align-items: center; justify-content: center; gap: var(--space-sm); flex-wrap: wrap; }
  figure { margin: 0; display: flex; flex-direction: column; align-items: center; gap: 6px; transition: opacity 250ms ease; }
  figure.pending { opacity: 0.28; }
  figcaption { font-size: 0.7rem; color: var(--ink-dim); transition: color 200ms; }
  figcaption.active { color: var(--accent); }
  .dim { color: var(--ink-faint); }
  svg { box-shadow: 0 0 0 1px var(--line); border-radius: 2px; image-rendering: pixelated; overflow: visible; }
  .win { fill: rgba(31,58,95,0.12); stroke: var(--accent); stroke-width: 2; vector-effect: non-scaling-stroke;
         transition: x 90ms linear, y 90ms linear; }
  .cur { fill: none; stroke: var(--accent); stroke-width: 2; vector-effect: non-scaling-stroke; }
  .conv { color: var(--ink-faint); font-size: 0.64rem; line-height: 1.1; text-align: center; transition: color 200ms; }
  .conv.active { color: var(--accent); }
  @media (max-width: 720px) { .conv { display: none; } }

  .rf-note { color: var(--ink-faint); font-size: 0.78rem; font-style: italic; text-align: center; }
  .rf-note em { color: var(--ink-dim); font-style: italic; }
</style>
