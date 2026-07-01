<script lang="ts">
  /** VectorStrip — a 512-d embedding drawn as a wrapped grid, one cell per dim. */
  import Heatmap from './Heatmap.svelte';

  type Props = { values: number[]; cols?: number; cell?: number; label?: string; color?: string };
  let { values, cols = 32, cell = 13, label, color = 'var(--accent)' }: Props = $props();

  const rows = $derived(Math.ceil(values.length / cols));
  const padded = $derived.by(() => {
    const n = rows * cols;
    if (values.length === n) return values;
    return [...values, ...Array(n - values.length).fill(NaN)];
  });
  const absMax = $derived.by(() => {
    let m = 0;
    for (const v of values) m = Math.max(m, Math.abs(v));
    return m || 1e-6;
  });
</script>

<div class="strip">
  {#if label}
    <div class="lbl mono" style:color>{label} <span class="dim">· {values.length}-d</span></div>
  {/if}
  <Heatmap values={padded} {rows} {cols} mode="div" {absMax} {cell} gap={1} />
</div>

<style>
  .strip { display: flex; flex-direction: column; gap: 6px; }
  .lbl { font-size: 0.8rem; }
  .dim { color: var(--ink-faint); }
</style>
