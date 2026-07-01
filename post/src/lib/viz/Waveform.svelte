<script lang="ts">
  type Props = { samples: number[]; height?: number; color?: string };
  let { samples, height = 80, color = 'var(--accent)' }: Props = $props();

  let canvas: HTMLCanvasElement;
  let wrap: HTMLDivElement;
  let width = $state(600);

  function paint() {
    if (!canvas || !samples.length) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const mid = height / 2;
    let peak = 0;
    for (const s of samples) peak = Math.max(peak, Math.abs(s));
    const norm = peak > 0 ? mid / peak : mid;

    ctx.strokeStyle = getComputedStyle(canvas).getPropertyValue('color') || color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < samples.length; i++) {
      const x = (i / (samples.length - 1)) * width;
      const y = mid - samples[i] * norm * 0.92;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.strokeStyle = 'var(--line)';
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(width, mid);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  $effect(() => {
    void [samples, width, height];
    paint();
  });

  $effect(() => {
    if (!wrap) return;
    const ro = new ResizeObserver((e) => {
      width = e[0].contentRect.width;
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  });
</script>

<div bind:this={wrap} class="wf" style:color>
  <canvas bind:this={canvas}></canvas>
</div>

<style>
  .wf { width: 100%; }
  canvas { display: block; width: 100%; }
</style>
