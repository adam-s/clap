<script lang="ts">
  /** Fixed bottom scrubber for the pipeline-stage playhead. */
  import { onMount, onDestroy } from 'svelte';
  import { STAGES, stageIdx, playing, setStage } from '../stores/clap';

  let cur = $state(0);
  let isPlaying = $state(false);
  const u1 = stageIdx.subscribe((v) => (cur = v));
  const u2 = playing.subscribe((v) => (isPlaying = v));

  function toggle() {
    if (!isPlaying && cur >= STAGES.length - 1) setStage(0);
    playing.update((p) => !p);
  }
  function next() { playing.set(false); setStage(cur + 1); }
  function prev() { playing.set(false); setStage(cur - 1); }

  function onKey(e: KeyboardEvent) {
    const el = e.target as HTMLElement;
    if (el && (el.tagName === 'INPUT' || el.isContentEditable)) return;
    if (e.key === ' ') { e.preventDefault(); toggle(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); next(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); prev(); }
  }
  onMount(() => window.addEventListener('keydown', onKey));
  onDestroy(() => { window.removeEventListener('keydown', onKey); u1(); u2(); });
</script>

<div class="bar">
  <div class="controls">
    <button onclick={prev} disabled={cur === 0} aria-label="Previous stage">‹</button>
    <button class="play" onclick={toggle} aria-label="Play/pause">{isPlaying ? '❚❚' : '▶'}</button>
    <button onclick={next} disabled={cur === STAGES.length - 1} aria-label="Next stage">›</button>
  </div>
  <div class="track">
    {#each STAGES as s, i (s.key)}
      <button
        class="dot"
        class:active={i === cur}
        class:done={i < cur}
        onclick={() => { playing.set(false); setStage(i); }}
        title={`${s.label} · ${s.sub}`}
      >
        <span class="tick"></span>
        <span class="name">{s.label}</span>
      </button>
    {/each}
  </div>
</div>

<style>
  .bar {
    position: fixed;
    left: 0; right: 0; bottom: 0;
    z-index: 50;
    display: flex;
    align-items: center;
    gap: var(--space-md);
    padding: 8px max(16px, calc((100vw - var(--maxw)) / 2 + 16px));
    background: rgba(14, 17, 22, 0.9);
    backdrop-filter: blur(8px);
    border-top: 1px solid var(--line);
  }
  .controls { display: flex; gap: 4px; }
  button {
    background: var(--panel-2);
    color: var(--ink);
    border: 1px solid var(--line);
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.85rem;
  }
  .controls button { width: 34px; height: 30px; }
  .play { color: var(--accent); }
  button:disabled { opacity: 0.4; cursor: default; }
  .track {
    display: flex;
    flex: 1;
    gap: 2px;
    overflow-x: auto;
  }
  .dot {
    flex: 1;
    min-width: 54px;
    background: transparent;
    border: none;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 2px 0;
  }
  .tick {
    width: 100%;
    height: 4px;
    border-radius: 2px;
    background: var(--line);
  }
  .dot.done .tick { background: var(--accent-text); }
  .dot.active .tick { background: var(--accent); }
  .name {
    font-size: 0.64rem;
    color: var(--ink-faint);
    white-space: nowrap;
  }
  .dot.active .name { color: var(--ink); }
  @media (max-width: 820px) {
    .name { display: none; }
    .dot { min-width: 22px; }
  }
</style>
