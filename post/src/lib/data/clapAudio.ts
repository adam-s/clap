/** Browser audio decode → mono 16 kHz → centered window, for the live encoder. */

export const SR = 16000;

/** Decode an encoded audio buffer to mono Float32 at 16 kHz. */
export async function decodeMono16k(buf: ArrayBuffer): Promise<Float32Array> {
  const Ctx = (window.AudioContext || (window as any).webkitAudioContext);
  const tmp = new Ctx();
  const audio = await tmp.decodeAudioData(buf.slice(0));
  tmp.close();
  const OAC = (window.OfflineAudioContext || (window as any).webkitOfflineAudioContext);
  const frames = Math.max(1, Math.ceil((audio.duration * SR)));
  const oac = new OAC(1, frames, SR);
  const src = oac.createBufferSource();
  src.buffer = audio;
  src.connect(oac.destination);
  src.start();
  const rendered = await oac.startRendering();
  return rendered.getChannelData(0).slice();
}

/** Centered `winSec`-second window (zero-padded if shorter). */
export function centerWindow(samples: Float32Array, winSec: number): Float32Array {
  const win = Math.round(SR * winSec);
  if (samples.length <= win) {
    const out = new Float32Array(win);
    out.set(samples, 0);
    return out;
  }
  const start = (samples.length - win) >> 1;
  return samples.slice(start, start + win);
}

export async function decodeUrlToWindow(url: string, winSec: number): Promise<Float32Array> {
  const buf = await fetch(url).then((r) => r.arrayBuffer());
  return centerWindow(await decodeMono16k(buf), winSec);
}
