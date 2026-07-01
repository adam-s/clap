<script lang="ts">
  /**
   * SpectrogramStep — the waveform → picture conversion, with the Fourier
   * transform made visible. A slice glides along the wave; its real |FFT|²
   * spectrum (computed live) blooms as a magma frequency analyzer; the mel
   * filterbank pools it into one column; the columns stack into the image.
   */
  import { onDestroy, onMount } from 'svelte';
  import type { ClapBundle, Clip } from '../data/clapData';
  import { activeSamples } from '../data/clapLive.svelte';
  import { powerSpectrogram } from '../data/clapEngine';
  import { sequential } from './colormap';
  import Transport from '../components/Transport.svelte';
  import type { Transport as AudioTransport } from '../audio/transport.svelte';
  import KaTeX from '../components/Math.svelte';

  type Props = { bundle: ClapBundle; clip: Clip; transport: AudioTransport; n: number };
  let { bundle, clip, transport, n }: Props = $props();

  const N = $derived(bundle.meta.img); // 32 columns / mel bins
  const winSec = $derived(bundle.meta.win_sec ?? 1.5);

  const lo = $derived(Math.min(...clip.mel.data));
  const hi = $derived(Math.max(...clip.mel.data));
  function melCol(j: number): number[] {
    const out: number[] = [];
    for (let r = 0; r < N; r++) out.push(clip.mel.data[r * N + j]);
    return out;
  }

  // ── live power spectrogram of the actual window ─────────────────────────────
  const spect = $derived(
    $activeSamples ? powerSpectrogram($activeSamples, bundle.weights) : null,
  );
  // fixed log scale across the whole clip → bars don't jump scale frame-to-frame
  const gScale = $derived.by(() => {
    if (!spect) return { min: -14, max: 0 };
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < spect.power.length; i++) {
      const l = Math.log(spect.power[i] + 1e-6);
      if (l < mn) mn = l; if (l > mx) mx = l;
    }
    return { min: mn, max: mx };
  });

  // ── playhead ────────────────────────────────────────────────────────────────
  // Default: the slice glides silently and loops. The speaker plays the actual
  // audio for ONE window sweep through the SHARED transport (same clock as the
  // footer bar), and while anything is playing the sweep locks to that clock.
  let phase = $state(0);          // 0 .. N
  const col = $derived(Math.min(N - 1, Math.floor(phase)));
  let playing = $state(true);     // visual auto-sweep running (when audio is idle)
  let everHeard = $state(false);  // hide the "hear it" hint after first use
  let visible = $state(true);
  let root: HTMLElement;
  let raf = 0, lastT = 0;
  const MS_PER_COL = 150;

  const winStart = $derived(Math.max(0, (transport.duration - winSec) / 2)); // centred window

  function hearWindow() { everHeard = true; playing = true; transport.playRegion(winStart, winStart + winSec); }
  function toggleSound() { transport.playing ? transport.pause() : hearWindow(); }
  function seek(v: number) { transport.pause(); playing = false; phase = v; }

  function loop(t: number) {
    raf = requestAnimationFrame(loop);
    if (!visible) { lastT = t; return; }
    if (transport.playing) {
      // lock the sweep to the shared audio clock across the analysed window
      phase = Math.min(N, Math.max(0, (transport.t - winStart) / winSec) * N);
      lastT = t;
    } else if (playing) {
      if (!lastT) lastT = t;
      phase += (t - lastT) / MS_PER_COL; lastT = t;
      if (phase >= N) phase = 0;
    } else { lastT = t; }
  }
  $effect(() => { void clip; });
  onMount(() => {
    const io = new IntersectionObserver((e) => (visible = e[0].isIntersecting), { threshold: 0 });
    if (root) io.observe(root);
    raf = requestAnimationFrame(loop);
    return () => io.disconnect();
  });
  onDestroy(() => cancelAnimationFrame(raf));

  // spectrum bars at the current (interpolated) frame
  const NB = 96;
  const bars = $derived.by(() => {
    if (!spect) return [] as number[];
    const { power, nframes, nfreq } = spect;
    const frameF = (N > 1 ? Math.min(1, phase / (N - 1)) : 0) * (nframes - 1);
    const f0 = Math.floor(frameF), f1 = Math.min(nframes - 1, f0 + 1), fr = frameF - f0;
    const out = new Array(NB);
    for (let b = 0; b < NB; b++) {
      const a = Math.floor((b / NB) * nfreq), z = Math.max(a + 1, Math.floor(((b + 1) / NB) * nfreq));
      let mx = 0;
      for (let f = a; f < z; f++) {
        const p = power[f0 * nfreq + f] * (1 - fr) + power[f1 * nfreq + f] * fr;
        if (p > mx) mx = p;
      }
      out[b] = Math.log(mx + 1e-6);
    }
    return out;
  });

  // ── oscillogram geometry + gliding slice ────────────────────────────────────
  const VW = 760, VH = 90, MID = VH / 2;
  const wavePath = $derived.by(() => {
    const w = clip.waveform;
    if (w.length < 2) return '';   // guard: length-1 → /(w.length-1) is NaN
    let peak = 0; for (const v of w) peak = Math.max(peak, Math.abs(v));
    const norm = peak > 0 ? (VH / 2 - 4) / peak : 0;
    let d = '';
    for (let i = 0; i < w.length; i++) d += `${i ? 'L' : 'M'}${((i / (w.length - 1)) * VW).toFixed(1)} ${(MID - w[i] * norm).toFixed(1)} `;
    return d;
  });
  const winX = $derived((phase / N) * VW);
  const winW = $derived(VW / N);

  const FW = 300, FH = 108, bw = FW / NB;
  const SC = 7, STRIP = 9;
  const tex = String.raw`\htmlClass{part-mel}{\text{mel}}[m,\,\htmlClass{part-col}{j}] = \sum_{f} \underbrace{\big|\,\mathcal{F}(x\cdot w_{\htmlClass{part-col}{j}})[f]\,\big|^{2}}_{\htmlClass{part-spec}{\text{spectrum}}}\; \htmlClass{part-fb}{M[f,m]}`;
</script>

<section class="step" bind:this={root}>
  <header>
    <span class="eyebrow mono">{String(n).padStart(2, '0')} · Waveform → Spectrogram</span>
    <span class="dims mono">~{Math.round(bundle.meta.win_sec * bundle.meta.sample_rate / 1000)}k samples → {N}×{N} image</span>
  </header>

  <p class="explain">
    A sound is just air pressure over time: the wave below, which you can hear.
    But a raw wave is too long, and tells you <em>when</em> things happen, not
    <em>which frequencies</em> are present. So we slide a short slice along it and
    run a <em>Fourier transform</em>: it reports how much of each frequency the
    slice holds — a held tone spikes at one frequency, a click spreads across many.
    Pool that spectrum into {N} mel bands and you have one column. Stack the columns
    and the sound becomes the image the convolutions read.
  </p>

  <div class="formula"><KaTeX {tex} displayMode /></div>

  <!-- waveform with the gliding slice -->
  <figure class="wf">
    <svg viewBox={`0 0 ${VW} ${VH}`} preserveAspectRatio="none" width="100%" height={VH}>
      <line class="axis" x1="0" y1={MID} x2={VW} y2={MID} />
      <path class="wave" d={wavePath} />
      <rect class="slice" x={winX} y="0" width={winW} height={VH} />
    </svg>
  </figure>

  <div class="row">
    <!-- the live Fourier spectrum of this slice -->
    <figure class="spectrum">
      <figcaption class="mono spec">|FFT|² · frequencies in this slice</figcaption>
      <svg viewBox={`0 0 ${FW} ${FH}`} width={FW} height={FH}>
        {#each bars as v, i (i)}
          {@const t = gScale.max > gScale.min ? (v - gScale.min) / (gScale.max - gScale.min) : 0}
          <rect x={i * bw} y={FH - t * FH} width={bw - 0.4} height={t * FH} fill={sequential(v, gScale.min, gScale.max)} />
        {/each}
      </svg>
      <span class="ax mono">low → frequency → high</span>
    </figure>

    <div class="to mono">pooled →</div>

    <!-- the column this slice produces -->
    <figure class="strip">
      <figcaption class="mono col">column {col + 1}</figcaption>
      <svg viewBox={`0 0 ${STRIP * 2.4} ${N * STRIP}`} width={STRIP * 2.4} height={N * STRIP}>
        {#each melCol(col) as v, r (r)}
          <rect x="0" y={r * STRIP} width={STRIP * 2.4} height={STRIP} fill={sequential(v, lo, hi)} />
        {/each}
      </svg>
    </figure>

    <div class="to mono">stacks →</div>

    <!-- the spectrogram filling column by column -->
    <figure class="spec-fig">
      <figcaption class="mono mel">mel spectrogram · {N}×{N}</figcaption>
      <svg viewBox={`0 0 ${N * SC} ${N * SC}`} width={N * SC} height={N * SC}>
        {#each clip.mel.data as v, k (k)}
          {@const c = k % N}
          <rect x={c * SC} y={Math.floor(k / N) * SC} width={SC} height={SC}
            fill={c <= col ? sequential(v, lo, hi) : 'var(--panel-2)'} />
        {/each}
        <rect class="curcol" x={col * SC} y="0" width={SC} height={N * SC} />
      </svg>
    </figure>
  </div>

  <div class="controls">
    <Transport {playing} value={col} max={N - 1} label={`${col + 1}/${N}`}
      onToggle={() => (playing = !playing)} onSeek={seek} />
    <div class="sound">
      {#if !everHeard}
        <div class="cue" aria-hidden="true">
          <span class="hint mono">hear it</span>
          <svg class="hint-arrow" viewBox="0 0 50 40" width="50" height="40">
            <path d="M2 9 C 20 4, 38 10, 41 33" fill="none" stroke="var(--accent)" stroke-width="1.7" stroke-linecap="round" />
            <path d="M34 28 L 42 35 L 44 24" fill="none" stroke="var(--accent)" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </div>
      {/if}
      <button class="speaker" class:on={transport.playing} onclick={toggleSound}
        aria-label={transport.playing ? 'Stop sound' : 'Play this clip'} title={transport.playing ? 'Playing — click to mute' : 'Muted — click to hear this clip'}>
        <svg viewBox="0 0 24 24" width="20" height="20">
          <path d="M3 9.5v5h3.5L11 18V6L6.5 9.5H3z" fill="currentColor" />
          {#if transport.playing}
            <path d="M14.5 8.5q2.5 3.5 0 7" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" />
            <path d="M17 6.5q4 5.5 0 11" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" />
          {:else}
            <path d="M15 9.5l5 5M20 9.5l-5 5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
          {/if}
        </svg>
      </button>
    </div>
  </div>
</section>

<style>
  .step { display: flex; flex-direction: column; gap: 14px; padding: var(--space-xl) 0; border-top: 1px solid var(--line); }
  header { display: flex; align-items: baseline; gap: 14px; }
  .eyebrow { color: var(--ink-faint); font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .dims { color: var(--ink-faint); font-size: 0.72rem; margin-left: auto; }
  .explain { margin: 0; color: var(--ink-dim); font-size: 0.95rem; line-height: 1.6; max-width: 64ch; }
  .explain :global(em) { color: var(--ink); font-style: italic; }
  .formula { color: var(--ink); font-size: 0.95rem; }
  .formula :global(.part-spec), .spec { color: var(--data); }
  .mel { color: var(--data); }
  .formula :global(.part-fb) { color: var(--weight); }
  .formula :global(.part-col), .col { color: var(--accent); }

  .wf { margin: 0; }
  .wf svg { display: block; border: 1px solid var(--line); border-radius: 4px; background: var(--panel); width: 100%; }
  .axis { stroke: var(--line-strong); }
  .wave { fill: none; stroke: var(--x-color); stroke-width: 1; vector-effect: non-scaling-stroke; }
  .slice { fill: rgba(31,58,95,0.14); stroke: var(--accent); stroke-width: 1.5; vector-effect: non-scaling-stroke; }

  .row { display: flex; align-items: center; justify-content: center; gap: var(--space-md); flex-wrap: wrap; }
  figure { margin: 0; display: flex; flex-direction: column; align-items: center; gap: 6px; }
  figcaption { font-size: 0.7rem; color: var(--ink-dim); }
  .strip figcaption { width: 92px; text-align: center; }
  .spectrum svg { background: #14080f; border-radius: 4px; box-shadow: 0 0 0 1px var(--line); }
  .ax { font-size: 0.62rem; color: var(--ink-faint); }
  .strip svg, .spec-fig svg { box-shadow: 0 0 0 1px var(--line); border-radius: 2px; image-rendering: pixelated; }
  .curcol { fill: none; stroke: var(--accent); stroke-width: 1.5; }
  .to { color: var(--ink-faint); font-size: 0.74rem; white-space: nowrap; }
  @media (max-width: 720px) { .to { display: none; } }

  /* room on the right so the hint arrow + speaker never clip at the edge */
  .controls { display: flex; align-items: center; gap: var(--space-md); margin-top: 2px; padding-right: var(--space-sm); overflow: visible; }
  .controls > :global(.transport) { flex: 1; }
  .sound { position: relative; display: flex; align-items: center; overflow: visible; }
  /* the cue floats above the speaker; the arrow curves down onto it */
  .cue { position: absolute; right: 2px; bottom: 34px; display: flex; align-items: flex-end; gap: 2px; pointer-events: none; }
  .hint { color: var(--accent); font-size: 0.74rem; font-style: italic; white-space: nowrap; margin-bottom: 8px; }
  .hint-arrow { overflow: visible; animation: nudge 1.6s ease-in-out infinite; }
  @keyframes nudge { 0%, 100% { transform: translateX(0); } 50% { transform: translateX(3px); } }
  .speaker {
    width: 38px; height: 38px; border-radius: 50%;
    border: 1px solid var(--line-strong); background: var(--bg);
    color: var(--ink-dim); cursor: pointer;
    display: grid; place-items: center;
    transition: background 0.12s, border-color 0.12s, color 0.12s;
  }
  .speaker:hover { border-color: var(--accent); color: var(--accent); }
  .speaker.on { border-color: var(--accent); color: var(--accent); background: color-mix(in srgb, var(--accent) 10%, var(--bg)); }
  @media (prefers-reduced-motion: reduce) { .hint-arrow { animation: none; } }
</style>
