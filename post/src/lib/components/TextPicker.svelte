<script lang="ts">
  /** The fixed label set, as selectable chips — the text-side analogue of ClipPicker. */
  import type { TextItem } from '../data/clapData';

  type Props = { items: TextItem[]; selected: number; onpick: (i: number) => void };
  let { items, selected, onpick }: Props = $props();
</script>

<div class="chips">
  {#each items as it, i (it.slug)}
    <button class="chip mono" class:on={i === selected} onclick={() => onpick(i)}>
      {it.caption}
    </button>
  {/each}
</div>

<style>
  .chips { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 6px; }
  .chip { font-size: 0.84rem; padding: 7px 15px; border-radius: 999px; cursor: pointer;
    border: 1px solid var(--line-strong); background: var(--panel); color: var(--ink-dim); transition: all 120ms; }
  .chip:hover { border-color: var(--accent); color: var(--ink); }
  .chip.on { background: var(--accent); border-color: var(--accent); color: #fff; }
</style>
