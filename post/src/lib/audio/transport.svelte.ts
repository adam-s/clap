/**
 * The single playback clock for the whole page (mirrors ~/Projects/separate).
 *
 * One persistent <audio> element, created once and reused across clips. One rAF
 * loop drives the reactive `t` and a `frame` counter; every visual reads those and
 * redraws in lockstep, so the footer bar and step 01 can never drift or double-play.
 * A clip change just calls load() — the SAME transport object stays referenced
 * everywhere, so nothing remounts. `playRegion` plays just the analysed window.
 */
export class Transport {
  playing = $state(false);
  /** current playback position, seconds — read for declarative playheads */
  t = $state(0);
  duration = $state(0);
  /** bumped once per rAF tick while playing — depend on it for imperative redraws */
  frame = $state(0);

  #raf = 0;
  #audio: HTMLAudioElement | null = null;
  #stopAt: number | null = null;

  #ensureAudio(): HTMLAudioElement | null {
    if (this.#audio || typeof Audio === 'undefined') return this.#audio;
    const a = new Audio();
    a.preload = 'auto';
    a.addEventListener('ended', () => this.pause());
    a.addEventListener('error', () => this.pause());
    a.addEventListener('loadedmetadata', () => {
      if (this.#audio && isFinite(this.#audio.duration)) this.duration = this.#audio.duration;
    });
    this.#audio = a;
    return a;
  }

  /** Point the transport at a new clip. Reuses the same element. */
  load(audioUrl: string, duration = 0) {
    this.pause();
    this.duration = duration;
    this.t = 0;
    const a = this.#ensureAudio();
    if (a) { a.src = audioUrl; a.load(); }
  }

  toggle() { this.playing ? this.pause() : this.play(); }

  play() {
    if (this.playing) return;
    this.#stopAt = null;
    if (this.duration && this.t >= this.duration - 0.01) this.seek(0);
    this.playing = true;
    const a = this.#ensureAudio();
    if (a) { a.currentTime = this.t; void a.play().catch(() => this.pause()); }
    this.#tick();
  }

  /** Play only [start, end] then pause — step 01's analysed window. */
  playRegion(start: number, end: number) {
    this.pause();
    this.seek(start);
    this.#stopAt = end;
    this.playing = true;
    const a = this.#ensureAudio();
    if (a) { a.currentTime = start; void a.play().catch(() => this.pause()); }
    this.#tick();
  }

  pause() {
    this.playing = false;
    cancelAnimationFrame(this.#raf);
    this.#audio?.pause();
  }

  seek(t: number) {
    const clamped = Math.max(0, Math.min(this.duration || Infinity, t));
    this.t = clamped;
    if (this.#audio) this.#audio.currentTime = clamped;
  }

  #tick = () => {
    if (this.#audio) this.t = this.#audio.currentTime;
    this.frame++;
    if (this.#stopAt != null && this.t >= this.#stopAt) { this.#stopAt = null; this.pause(); return; }
    this.#raf = requestAnimationFrame(this.#tick);
  };

  destroy() {
    this.pause();
    if (this.#audio) this.#audio.src = '';
  }
}
