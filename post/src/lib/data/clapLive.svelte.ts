/**
 * clapLive — the live encoder state. Decodes a clip (a demo wav or an uploaded
 * file), runs the in-browser forward pass, and exposes the computed Clip so
 * every step renders real values flowing through actual audio. No precomputed
 * activations: the numbers are produced here, on demand.
 */
import { writable } from 'svelte/store';
import type { Clip, Weights } from './clapData';
import { forward } from './clapEngine';
import { decodeMono16k, centerWindow, SR } from './clapAudio';

export const activeClip = writable<Clip | null>(null);
export const activeSamples = writable<Float32Array | null>(null); // raw window, for live FFT
export const audioUrl = writable<string>('');
export const computing = writable<boolean>(false);
export const clipPeaks = writable<[number, number][]>([]); // full-clip waveform, for the footer

/** min/max envelope of the whole clip, for the footer waveform + playhead. */
function peaks(samples: Float32Array, n = 240): [number, number][] {
  const out: [number, number][] = [];
  const step = Math.max(1, Math.floor(samples.length / n));
  for (let i = 0; i < samples.length; i += step) {
    let mn = 0, mx = 0;
    for (let j = i; j < i + step && j < samples.length; j++) {
      const v = samples[j];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    out.push([mn, mx]);
  }
  return out;
}

let weights: Weights | null = null;
let demo: Clip[] = [];
let winSec = 2.0;

export function initLive(W: Weights, clips: Clip[], ws: number): void {
  weights = W;
  demo = clips;
  winSec = ws;
}

function waveformPreview(samples: Float32Array, n = 800): number[] {
  const step = Math.max(1, Math.floor(samples.length / n));
  const out: number[] = [];
  for (let i = 0; i < samples.length; i += step) out.push(Math.round(samples[i] * 1e4) / 1e4);
  return out;
}

async function run(samples: Float32Array, meta: { slug: string; file: string; caption: string }): Promise<void> {
  if (!weights) return;
  const fwd = forward(samples, weights);
  activeSamples.set(samples);
  clipPeaks.set(peaks(samples));         // the analysed window — same length for every clip
  activeClip.set({ ...meta, waveform: waveformPreview(samples), ...fwd });
}

export async function selectDemo(i: number): Promise<void> {
  const c = demo[i];
  if (!c || !weights) return;
  computing.set(true);
  try {
    const url = `${import.meta.env.BASE_URL}audio/${c.file}`;
    audioUrl.set(url);
    const buf = await fetch(url).then((r) => r.arrayBuffer());
    const samples = centerWindow(await decodeMono16k(buf), winSec);
    await run(samples, { slug: c.slug, file: c.file, caption: c.caption });
  } finally {
    computing.set(false);
  }
}

export async function loadFile(file: File): Promise<void> {
  if (!weights) return;
  computing.set(true);
  try {
    audioUrl.set(URL.createObjectURL(file));
    const samples = centerWindow(await decodeMono16k(await file.arrayBuffer()), winSec);
    await run(samples, { slug: 'your clip', file: file.name, caption: file.name.replace(/\.[^.]+$/, '') });
  } finally {
    computing.set(false);
  }
}

export { SR };
