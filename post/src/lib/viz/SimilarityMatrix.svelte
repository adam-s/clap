<script lang="ts">
  /**
   * SimilarityMatrix — the contrastive payoff: cosine(E^a_i, E^t_j) for every
   * audio×caption pair. The bright diagonal is each car sound matching its own
   * diagnosis caption. Real captured values; the diagonal is what training built.
   */
  import { diverging } from './colormap';

  type Props = {
    matrix: number[][];
    rowLabels: string[];
    colLabels: string[];
    highlightRow?: number | null;
    onpick?: (i: number) => void;
  };
  let { matrix, rowLabels, colLabels, highlightRow = null, onpick }: Props = $props();

  const absMax = $derived.by(() => {
    let m = 0;
    for (const row of matrix) for (const v of row) m = Math.max(m, Math.abs(v));
    return m || 1e-6;
  });
  const n = $derived(matrix.length);
</script>

<div class="wrap">
  <table>
    <thead>
      <tr>
        <th class="corner"></th>
        {#each colLabels as c (c)}
          <th class="col"><span>{c}</span></th>
        {/each}
      </tr>
    </thead>
    <tbody>
      {#each matrix as row, i (i)}
        <tr
          class:hl={highlightRow === i}
          onclick={() => onpick?.(i)}
          class:clickable={!!onpick}
        >
          <th class="row mono">{rowLabels[i]}</th>
          {#each row as v, j (j)}
            <td
              class:diag={i === j}
              style:background={diverging(v, absMax)}
              title={`${rowLabels[i]} ↔ ${colLabels[j]} = ${v.toFixed(2)}`}
            >
              <span class="mono">{v.toFixed(2)}</span>
            </td>
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
  .wrap { overflow-x: auto; }
  table { border-collapse: collapse; }
  th, td { padding: 0; }
  td {
    width: 46px;
    height: 38px;
    text-align: center;
    border: 1px solid var(--bg);
  }
  td span { font-size: 0.72rem; color: rgba(14, 17, 22, 0.82); font-weight: 600; }
  td.diag { outline: 2px solid var(--ink); outline-offset: -2px; }
  th.col {
    vertical-align: bottom;
    height: 90px;
    font-size: 0.72rem;
    color: var(--ink-dim);
    font-weight: 500;
  }
  th.col span {
    display: inline-block;
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    white-space: nowrap;
    padding-bottom: 4px;
  }
  th.row {
    text-align: right;
    padding-right: 8px;
    font-size: 0.74rem;
    color: var(--ink-dim);
    font-weight: 500;
    white-space: nowrap;
  }
  tr.clickable { cursor: pointer; }
  tr.hl th.row { color: var(--accent); }
  tr.hl td { filter: brightness(1.08); }
</style>
