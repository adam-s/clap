<script lang="ts">
  /**
   * Transport — the one play/scrub control used by every step (and reused for
   * audio). Round play button + a slim seek bar joined as a single unit, so the
   * iconography and behaviour stay consistent with the other adamsohn.com posts
   * (grpo / separate): ▶ to play, ⏸ to pause, a thin accent-coloured scrubber.
   */
  type Props = {
    playing: boolean;
    value: number;
    max: number;
    step?: number;
    label?: string;
    onToggle: () => void;
    onSeek: (v: number) => void;
  };
  let { playing, value, max, step = 1, label, onToggle, onSeek }: Props = $props();
</script>

<div class="transport">
  <button class="play" aria-label={playing ? 'Pause' : 'Play'} onclick={onToggle}>
    {playing ? '⏸' : '▶'}
  </button>
  <input
    class="scrub"
    type="range"
    min="0"
    {max}
    {step}
    {value}
    aria-label="Seek"
    oninput={(e) => onSeek(+(e.target as HTMLInputElement).value)}
  />
  {#if label}<span class="lbl mono">{label}</span>{/if}
</div>

<style>
  .transport {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
  }
  .play {
    flex: none;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    border: 1px solid var(--line-strong);
    background: var(--bg);
    color: var(--accent);
    font-size: 0.7rem;
    line-height: 1;
    cursor: pointer;
    display: grid;
    place-items: center;
    padding: 0;
    transition: background 0.12s ease, border-color 0.12s ease;
  }
  .play:hover { background: var(--panel); border-color: var(--accent); }
  .play:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .scrub {
    flex: 1;
    height: 3px;
    accent-color: var(--accent);
    cursor: pointer;
  }
  .lbl {
    flex: none;
    font-size: 0.72rem;
    color: var(--ink-faint);
    min-width: 3ch;
    text-align: right;
  }
</style>
