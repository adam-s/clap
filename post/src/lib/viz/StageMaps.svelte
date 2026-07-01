<script lang="ts">
  /**
   * StageMaps — draws EVERY channel of a [H,W,C] activation map as a grid of
   * small HxW heatmaps. This is the exhaustive part: stage3 (2x2x64) shows all
   * 64 channels, nothing pooled or sampled. Fixed diverging scale per stage.
   */
  import Heatmap from './Heatmap.svelte';
  import { channel, type TensorMap } from '../data/clapData';

  type Props = { map: TensorMap; cell?: number; maxChannels?: number };
  let { map, cell = 9, maxChannels = 64 }: Props = $props();

  const rows = $derived(map.shape[0]);
  const cols = $derived(map.shape[1]);
  const C = $derived(map.shape[2]);
  const shown = $derived(Math.min(C, maxChannels));

  const absMax = $derived.by(() => {
    let m = 0;
    for (const v of map.data) m = Math.max(m, Math.abs(v));
    return m || 1e-6;
  });

  const channels = $derived(
    Array.from({ length: shown }, (_, c) => channel(map, c)),
  );
</script>

<div class="stage-maps">
  {#each channels as ch, i (i)}
    <div class="chan">
      <Heatmap values={ch} {rows} {cols} mode="div" {absMax} {cell} gap={0} />
    </div>
  {/each}
</div>
{#if C > shown}
  <div class="trunc mono">showing {shown} of {C} channels</div>
{/if}

<style>
  .stage-maps {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: flex-start;
  }
  .chan {
    padding: 2px;
    background: var(--panel-2);
    border-radius: 3px;
    line-height: 0;
  }
  .trunc { color: var(--ink-faint); font-size: 0.72rem; margin-top: 6px; }
</style>
