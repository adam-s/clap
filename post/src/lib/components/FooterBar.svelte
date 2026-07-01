<script lang="ts">
  /**
   * FooterBar — switch the sound and play it while you scroll. It plays the same
   * 1.5s window the encoder analyses (so every clip is the same length and step 01's
   * sweep tracks it exactly), through the ONE shared transport. While the sound
   * plays, the footer + step 01 are in lockstep; when it's stopped the footer sits
   * at the beginning and the silent demo sweep runs. Mirrors ~/Projects/separate.
   */
  import type { Clip } from '../data/clapData';
  import type { Transport } from '../audio/transport.svelte';
  import { activeClip, clipPeaks, selectDemo } from '../data/clapLive.svelte';

  type Props = {
    clips: Pick<Clip, 'slug' | 'file' | 'caption'>[];
    transport: Transport;
    winSec: number;
  };
  let { clips, transport, winSec }: Props = $props();

  let canvas: HTMLCanvasElement;
  const W = 200, H = 34;

  // the analysed window is centred in each clip; its length is fixed for every clip
  const winStart = $derived(Math.max(0, (transport.duration - winSec) / 2));
  const winLen = $derived(Math.min(winSec, transport.duration || winSec));
  // position within the window; 0 whenever the sound is stopped
  const pos = $derived(transport.playing ? Math.max(0, Math.min(winLen, transport.t - winStart)) : 0);

  const fmt = (s: number) => `${Math.max(0, s).toFixed(1)}s`;

  function toggle() {
    if (transport.playing) transport.pause();
    else transport.playRegion(winStart, winStart + winLen);
  }

  function draw() {
    const pk = $clipPeaks;
    if (!canvas || !pk.length) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (canvas.width !== W * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; }
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const mid = H / 2;
    const px = (winLen ? pos / winLen : 0) * W;
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x++) {
      const [mn, mx] = pk[Math.floor((x / W) * pk.length)] ?? [0, 0];
      ctx.strokeStyle = x <= px ? '#b8551f' : '#c8c1b6'; // played warm, rest muted
      ctx.beginPath();
      ctx.moveTo(x + 0.5, mid + mn * (H / 2 - 1));
      ctx.lineTo(x + 0.5, mid + mx * (H / 2 - 1));
      ctx.stroke();
    }
    ctx.strokeStyle = '#1f3a5f';
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
  }

  // redraw as the clock ticks (playhead) or the clip changes (new waveform)
  $effect(() => { void pos; void $clipPeaks; draw(); });
</script>

<footer class="bar">
  <select
    class="switch mono"
    value={$activeClip?.slug ?? ''}
    onchange={(e) => {
      const i = clips.findIndex((c) => c.slug === (e.target as HTMLSelectElement).value);
      if (i >= 0) selectDemo(i);
    }}
    aria-label="Choose a recording"
  >
    {#each clips as c (c.slug)}
      <option value={c.slug}>{c.caption}</option>
    {/each}
    {#if $activeClip && !clips.some((c) => c.slug === $activeClip?.slug)}
      <option value={$activeClip.slug}>{$activeClip.caption}</option>
    {/if}
  </select>

  <button class="play" aria-label={transport.playing ? 'Pause' : 'Play'} onclick={toggle}>
    {transport.playing ? '❚❚' : '▶'}
  </button>

  <input
    class="scrub"
    type="range"
    min="0"
    max={winLen || 1}
    step="0.01"
    value={pos}
    aria-label="Seek"
    oninput={(e) => transport.seek(winStart + +(e.target as HTMLInputElement).value)}
  />
  <span class="time mono">{fmt(pos)} / {fmt(winLen)}</span>
  <canvas bind:this={canvas} style:width={`${W}px`} style:height={`${H}px`} aria-hidden="true"></canvas>
</footer>

<style>
  .bar {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 40;
    display: flex; align-items: center; gap: var(--space-md);
    padding: 8px max(var(--space-lg), env(safe-area-inset-left)) calc(8px + env(safe-area-inset-bottom));
    background: rgba(253, 252, 249, 0.9);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-top: 1px solid var(--line-strong);
  }
  .switch {
    max-width: 220px; flex: none;
    font-size: 0.8rem; color: var(--ink);
    background: var(--bg); border: 1px solid var(--line-strong);
    border-radius: 8px; padding: 6px 8px; cursor: pointer;
  }
  .play {
    flex: none; width: 34px; height: 34px; border-radius: 50%;
    border: 1px solid var(--line-strong); background: var(--bg); color: var(--ink);
    font-size: 0.62rem; cursor: pointer; display: grid; place-items: center;
  }
  .play:hover { border-color: var(--accent); color: var(--accent); }
  .scrub { flex: 1; min-width: 40px; accent-color: var(--accent); cursor: pointer; }
  .time { flex: none; font-size: 0.72rem; color: var(--ink-faint); font-variant-numeric: tabular-nums; white-space: nowrap; }
  canvas { flex: none; display: block; border: 1px solid var(--line); border-radius: 3px; background: var(--panel); }

  @media (max-width: 560px) {
    .bar { gap: var(--space-sm); padding: 7px var(--space-md) calc(7px + env(safe-area-inset-bottom)); }
    .time, canvas { display: none; }
    .switch { max-width: 150px; }
  }
</style>
