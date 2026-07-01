<script lang="ts">
  /**
   * Heatmap — draws a rows×cols grid of real values as cells on a canvas, one
   * cell per value. The pixel-perfect primitive: nothing is summarized, the
   * grid dimensions ARE the tensor dimensions.
   */
  import { diverging, sequential } from './colormap';

  type Props = {
    values: number[];
    rows: number;
    cols: number;
    mode?: 'div' | 'seq';
    /** fixed scale for 'div'; if absent, computed from |values|. */
    absMax?: number;
    cell?: number;
    gap?: number;
    grid?: boolean;
  };
  let {
    values,
    rows,
    cols,
    mode = 'div',
    absMax,
    cell = 14,
    gap = 1,
    grid = false,
  }: Props = $props();

  let canvas: HTMLCanvasElement;

  const w = $derived(cols * cell + (cols - 1) * gap);
  const h = $derived(rows * cell + (rows - 1) * gap);

  function paint() {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    let amax = absMax ?? 0;
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of values) {
      if (!Number.isFinite(v)) continue;
      if (mode === 'div' && absMax == null) amax = Math.max(amax, Math.abs(v));
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (amax === 0) amax = 1e-6;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = values[r * cols + c];
        if (!Number.isFinite(v)) continue;
        ctx.fillStyle = mode === 'div' ? diverging(v, amax) : sequential(v, lo, hi);
        ctx.fillRect(c * (cell + gap), r * (cell + gap), cell, cell);
      }
    }
    if (grid && cell >= 6) {
      ctx.strokeStyle = 'rgba(0,0,0,0.25)';
      ctx.lineWidth = 0.5;
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++)
          ctx.strokeRect(c * (cell + gap) + 0.25, r * (cell + gap) + 0.25, cell, cell);
    }
  }

  $effect(() => {
    // re-read reactive deps so paint runs on any change
    void [values, rows, cols, mode, absMax, cell, gap, grid];
    paint();
  });
</script>

<canvas bind:this={canvas} class="heatmap"></canvas>

<style>
  .heatmap {
    display: block;
    border-radius: 3px;
    image-rendering: pixelated;
  }
</style>
