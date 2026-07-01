/**
 * clapEngine.ts — the miniature CLAP audio encoder, reimplemented in TypeScript
 * so a real clip's values can be computed live in the browser and flow through
 * every step. Mirrors toy/clap_toy.py exactly:
 *   samples → log-mel (STFT + mel filterbank + standardize + resize 32×32)
 *           → 4× conv(3×3, stride 2) + ReLU → global avg-pool(64)
 *           → Linear(64→512) ReLU Linear(512→512) → L2-normalize → 512.
 *
 * Outputs the same shapes the capture dumps, so the viz is agnostic to whether
 * a clip is precomputed JSON or computed here.
 */
import type { Clip, TensorMap, Weights } from './clapData';

// ── FFT (iterative radix-2, power-of-two) ─────────────────────────────────────
function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { const tr = re[i]; re[i] = re[j]; re[j] = tr; const ti = im[i]; im[i] = im[j]; im[j] = ti; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len >> 1; k++) {
        const ar = re[i + k], ai = im[i + k];
        const br = re[i + k + (len >> 1)], bi = im[i + k + (len >> 1)];
        const vr = br * cr - bi * ci, vi = br * ci + bi * cr;
        re[i + k] = ar + vr; im[i + k] = ai + vi;
        re[i + k + (len >> 1)] = ar - vr; im[i + k + (len >> 1)] = ai - vi;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

function hann(n: number): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n); // periodic
  return w;
}

/** torch reflect-pad (no edge repeat) by `p` on both sides. */
function reflectPad(x: Float32Array, p: number): Float64Array {
  const n = x.length, out = new Float64Array(n + 2 * p);
  for (let i = 0; i < n; i++) out[p + i] = x[i];
  for (let i = 0; i < p; i++) { out[p - 1 - i] = x[i + 1]; out[p + n + i] = x[n - 2 - i]; }
  return out;
}

/** bilinear resize a [h0,w0] row-major grid to [h1,w1], align_corners=false. */
function resize(src: number[], h0: number, w0: number, h1: number, w1: number): number[] {
  const out = new Array(h1 * w1);
  const sample = (r: number, c: number) => src[r * w0 + c];
  for (let yo = 0; yo < h1; yo++) {
    let sy = ((yo + 0.5) * h0) / h1 - 0.5; sy = Math.max(0, Math.min(h0 - 1, sy));
    const y0 = Math.floor(sy), y1 = Math.min(h0 - 1, y0 + 1), fy = sy - y0;
    for (let xo = 0; xo < w1; xo++) {
      let sx = ((xo + 0.5) * w0) / w1 - 0.5; sx = Math.max(0, Math.min(w0 - 1, sx));
      const x0 = Math.floor(sx), x1 = Math.min(w0 - 1, x0 + 1), fx = sx - x0;
      const a = sample(y0, x0), b = sample(y0, x1), c = sample(y1, x0), d = sample(y1, x1);
      out[yo * w1 + xo] = a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + c * fy * (1 - fx) + d * fy * fx;
    }
  }
  return out;
}

/** Full power spectrogram (|STFT|²) — what the mel filterbank pools from.
 * Returns power [nframes × nfreq] row-major, for the live Fourier animation. */
export function powerSpectrogram(samples: Float32Array, W: Weights): { power: Float64Array; nframes: number; nfreq: number } {
  const p = W.mel;
  const nfft = p.n_fft, hop = p.hop, nfreq = nfft / 2 + 1;
  const win = hann(nfft);
  const padded = reflectPad(samples, nfft / 2);
  const nframes = 1 + Math.floor((padded.length - nfft) / hop);
  const power = new Float64Array(nframes * nfreq);
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  for (let t = 0; t < nframes; t++) {
    const off = t * hop;
    for (let i = 0; i < nfft; i++) { re[i] = padded[off + i] * win[i]; im[i] = 0; }
    fft(re, im);
    for (let f = 0; f < nfreq; f++) power[t * nfreq + f] = re[f] * re[f] + im[f] * im[f];
  }
  return { power, nframes, nfreq };
}

/** samples → standardized log-mel image, returned as a [IMG,IMG,1] TensorMap. */
export function melImage(samples: Float32Array, W: Weights): TensorMap {
  const p = W.mel;
  const nfft = p.n_fft, hop = p.hop, nmels = p.n_mels, img = p.resize;
  const nfreq = nfft / 2 + 1;
  const win = hann(nfft);
  const padded = reflectPad(samples, nfft / 2);
  const nframes = 1 + Math.floor((padded.length - nfft) / hop);
  const fb = W.melfb.data; // [nfreq, nmels] row-major
  const melTF: number[] = new Array(nmels * nframes); // [mel, frame]
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  for (let t = 0; t < nframes; t++) {
    const off = t * hop;
    for (let i = 0; i < nfft; i++) { re[i] = padded[off + i] * win[i]; im[i] = 0; }
    fft(re, im);
    // power spectrum, then mel = fbᵀ · power
    for (let m = 0; m < nmels; m++) {
      let s = 0;
      for (let f = 0; f < nfreq; f++) s += fb[f * nmels + m] * (re[f] * re[f] + im[f] * im[f]);
      melTF[m * nframes + t] = s;
    }
  }
  // log + standardize
  let mean = 0;
  for (let i = 0; i < melTF.length; i++) { melTF[i] = Math.log(melTF[i] + p.log_eps); mean += melTF[i]; }
  mean /= melTF.length;
  let v = 0;
  for (let i = 0; i < melTF.length; i++) v += (melTF[i] - mean) ** 2;
  const std = Math.sqrt(v / melTF.length) + 1e-5;
  for (let i = 0; i < melTF.length; i++) melTF[i] = (melTF[i] - mean) / std;
  // resize [nmels, nframes] → [img, img]
  const resized = resize(melTF, nmels, nframes, img, img);
  return { shape: [img, img, 1], data: resized };
}

// ── conv(3×3, stride 2, pad 1) + ReLU. in/out are [H,W,C] flat. ───────────────
function conv(inp: TensorMap, kernel: TensorMap, bias: TensorMap): TensorMap {
  const [Hi, Wi, Cin] = inp.shape;
  const [Cout] = kernel.shape; // [Cout, Cin, 3, 3]
  const Ho = Hi >> 1, Wo = Wi >> 1;
  const out = new Array(Ho * Wo * Cout).fill(0);
  for (let oc = 0; oc < Cout; oc++) {
    const b = bias.data[oc];
    for (let oi = 0; oi < Ho; oi++)
      for (let oj = 0; oj < Wo; oj++) {
        let s = b;
        for (let ic = 0; ic < Cin; ic++)
          for (let u = 0; u < 3; u++)
            for (let w = 0; w < 3; w++) {
              const ii = 2 * oi + u - 1, jj = 2 * oj + w - 1;
              if (ii < 0 || jj < 0 || ii >= Hi || jj >= Wi) continue;
              s += inp.data[(ii * Wi + jj) * Cin + ic] *
                kernel.data[((oc * Cin + ic) * 3 + u) * 3 + w];
            }
        out[(oi * Wo + oj) * Cout + oc] = s > 0 ? s : 0; // ReLU
      }
  }
  return { shape: [Ho, Wo, Cout], data: out };
}

function globalAvgPool(m: TensorMap): number[] {
  const [H, W, C] = m.shape;
  const out = new Array(C).fill(0);
  for (let i = 0; i < H * W; i++) for (let c = 0; c < C; c++) out[c] += m.data[i * C + c];
  for (let c = 0; c < C; c++) out[c] /= H * W;
  return out;
}

/** y = W·x + b, with optional ReLU. W is [out,in] row-major. */
function linear(x: number[], Wm: TensorMap, bias: TensorMap, relu: boolean): number[] {
  const [out, inn] = Wm.shape;
  const y = new Array(out);
  for (let o = 0; o < out; o++) {
    let s = bias.data[o];
    for (let i = 0; i < inn; i++) s += Wm.data[o * inn + i] * x[i];
    y[o] = relu && s < 0 ? 0 : s;
  }
  return y;
}

function l2norm(x: number[]): number[] {
  let n = 0; for (const v of x) n += v * v;
  n = Math.sqrt(n) || 1;
  return x.map((v) => v / n);
}

export type AudioForward = Pick<Clip, 'mel' | 'stages' | 'latent' | 'audioEmbed'>;

/** Full audio-encoder forward pass on raw samples. */
export function forward(samples: Float32Array, W: Weights): AudioForward {
  const mel = melImage(samples, W);
  const patch = conv(mel, W.conv.patch, W.convBias.patch);
  const stage1 = conv(patch, W.conv.stage1, W.convBias.stage1);
  const stage2 = conv(stage1, W.conv.stage2, W.convBias.stage2);
  const stage3 = conv(stage2, W.conv.stage3, W.convBias.stage3);
  const latent = globalAvgPool(stage3);
  const h = linear(latent, W.proj1, W.proj1Bias, true);
  const embed = l2norm(linear(h, W.proj2, W.proj2Bias, false));
  return {
    mel,
    stages: { patch, stage1, stage2, stage3 },
    latent: { shape: [latent.length], data: latent },
    audioEmbed: embed,
  };
}
