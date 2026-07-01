/**
 * clapData.ts — loads the captured run (public/data/*.json) and types it.
 * Everything is a real tensor from one forward pass of a miniature CLAP encoder
 * on a real ~2 s audio clip; the browser only draws. Mirrors grpo's
 * trajectoryData.ts (cached singleton fetch).
 */

export type TensorMap = { shape: number[]; data: number[] };

export type Clip = {
  slug: string;
  file: string;
  caption: string;
  waveform: number[];
  mel: TensorMap; // [IMG, IMG, 1]
  stages: {
    patch: TensorMap; // [16,16,8]
    stage1: TensorMap; // [8,8,16]
    stage2: TensorMap; // [4,4,32]
    stage3: TensorMap; // [2,2,64]
  };
  latent: TensorMap; // [64]
  audioEmbed: number[]; // [512]
};

export type TextItem = {
  slug: string;
  caption: string;
  tokens: string[]; // the words
  ids: number[]; // vocab ids of those words
  rawEmbed: TensorMap; // [T,64] embedding-table lookup (pre-context)
  tokenEmbed: TensorMap; // [T,64] after context mixing
  latent: number[]; // [64] masked-mean pooled
  textEmbed: number[]; // [512] projected + L2-normalised
};

export type StageMeta = { name: string; shape: number[]; note: string };

export type Meta = {
  schema: number;
  model: string;
  embed_dim: number;
  latent_dim: number;
  mel_bins: number;
  img: number;
  sample_rate: number;
  win_sec: number;
  stages: StageMeta[];
  vocab: Record<string, number>;
  logit_scale: number;
};

export type Similarity = {
  slugs: string[];
  captions: string[];
  cosine: number[][];
  probs: number[][];
};

export type Trajectory = {
  slugs: string[];
  captions: string[];
  snapshots: { step: number; sim: number[][] }[];
};

export type MelParams = {
  n_fft: number; hop: number; win: number; window: string; center: boolean;
  power: number; n_mels: number; f_min: number; f_max: number;
  sample_rate: number; log_eps: number; standardize: boolean; resize: number;
};

export type Weights = {
  conv: Record<'patch' | 'stage1' | 'stage2' | 'stage3', TensorMap>; // [out,in,3,3]
  convBias: Record<'patch' | 'stage1' | 'stage2' | 'stage3', TensorMap>;
  proj1: TensorMap; // [512,64]
  proj1Bias: TensorMap;
  proj2: TensorMap; // [512,512]
  proj2Bias: TensorMap;
  melfb: TensorMap; // [n_freqs, n_mels]
  embTable: TensorMap; // [vocab, 64] word → row
  mel: MelParams;
  stride: number;
  kernel: number;
};

export type ClapBundle = {
  meta: Meta;
  clips: Clip[];
  weights: Weights;
  text: TextItem[];
  similarity: Similarity;
};

let _cache: Promise<ClapBundle> | null = null;

async function getJSON<T>(name: string): Promise<T> {
  const resp = await fetch(`${import.meta.env.BASE_URL}data/${name}`);
  if (!resp.ok) throw new Error(`${name}: ${resp.status}`);
  return (await resp.json()) as T;
}

export function loadClap(): Promise<ClapBundle> {
  if (_cache) return _cache;
  _cache = Promise.all([
    getJSON<Meta>('meta.json'),
    getJSON<Clip[]>('clips.json'),
    getJSON<Weights>('weights.json'),
    getJSON<TextItem[]>('text.json'),
    getJSON<Similarity>('similarity.json'),
  ]).then(([meta, clips, weights, text, similarity]) => ({ meta, clips, weights, text, similarity }));
  return _cache;
}

/** Pull one channel (channel-last flat) out of a [H,W,C] map as a row-major H*W array. */
export function channel(map: TensorMap, c: number): number[] {
  const [h, w, ch] = map.shape;
  const out = new Array(h * w);
  for (let i = 0; i < h * w; i++) out[i] = map.data[i * ch + c];
  return out;
}

/** Mean over channels of a [H,W,C] map → H*W array (the faithful spatial summary). */
export function channelMean(map: TensorMap): number[] {
  const [h, w, ch] = map.shape;
  const out = new Array(h * w).fill(0);
  for (let i = 0; i < h * w; i++) {
    let s = 0;
    for (let c = 0; c < ch; c++) s += map.data[i * ch + c];
    out[i] = s / ch;
  }
  return out;
}
