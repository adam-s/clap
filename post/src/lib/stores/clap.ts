/**
 * clap.ts — cursors for the replay.
 *
 * Unlike grpo (whose playhead walked *training steps*), CLAP inference is one
 * forward pass, so the playhead walks *pipeline stages* — the journey from
 * waveform to the 512-d embedding. A second cursor picks which clip is flowing
 * through. Every figure subscribes to these.
 */
import { writable, get } from 'svelte/store';

/** Ordered stages of the audio tower — the playhead walks these. */
export const STAGES = [
  { key: 'waveform', label: 'Waveform', sub: '~2 s @ 16 kHz' },
  { key: 'mel', label: 'Mel image', sub: '32 × 32' },
  { key: 'patch', label: 'Patch embed', sub: '16 × 16 × 8' },
  { key: 'stage1', label: 'Stage 1', sub: '8 × 8 × 16' },
  { key: 'stage2', label: 'Stage 2', sub: '4 × 4 × 32' },
  { key: 'stage3', label: 'Stage 3', sub: '2 × 2 × 64' },
  { key: 'latent', label: 'Pooled', sub: '64' },
  { key: 'embed', label: 'Embedding', sub: '512' },
] as const;

export type StageKey = (typeof STAGES)[number]['key'];

/** Which clip is flowing through the pipeline. */
export const clipIdx = writable<number>(0);

/** Where along the pipeline the playhead sits (index into STAGES). */
export const stageIdx = writable<number>(0);

export const playing = writable<boolean>(false);

const PLAY_MS = 1100;
let timer: ReturnType<typeof setInterval> | null = null;
playing.subscribe((p) => {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
  if (!p) return;
  timer = setInterval(() => {
    const cur = get(stageIdx);
    if (cur >= STAGES.length - 1) {
      playing.set(false);
      return;
    }
    stageIdx.set(cur + 1);
  }, PLAY_MS);
});

export function setStage(i: number) {
  stageIdx.set(Math.max(0, Math.min(STAGES.length - 1, i)));
}
