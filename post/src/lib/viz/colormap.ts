/** Tiny colormaps for the cell renderers. Inputs are plain numbers. */

type RGB = [number, number, number];

function lerp(a: RGB, b: RGB, t: number): RGB {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function ramp(stops: RGB[], t: number): RGB {
  t = Number.isFinite(t) ? Math.max(0, Math.min(1, t)) : 0;   // NaN → 0, never index out of range
  const x = t * (stops.length - 1);
  const i = Math.floor(x);
  if (i >= stops.length - 1) return stops[stops.length - 1];
  return lerp(stops[i], stops[i + 1], x - i);
}

const css = (c: RGB) => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;

// Diverging blue–white–red (RdBu reversed), for signed activations / weights.
const DIV: RGB[] = [
  [33, 102, 172], [67, 147, 195], [146, 197, 222], [209, 229, 240],
  [247, 247, 247], [253, 219, 199], [244, 165, 130], [214, 96, 77], [178, 24, 43],
];

// Sequential magma-ish, for non-negative spectrogram energy.
const SEQ: RGB[] = [
  [12, 8, 38], [60, 15, 95], [122, 28, 109], [186, 54, 85], [232, 104, 57],
  [251, 176, 86], [252, 245, 180],
];

/** value in roughly [-absMax, absMax] → diverging color. */
export function diverging(v: number, absMax: number): string {
  const t = absMax > 0 ? (v / absMax + 1) / 2 : 0.5;
  return css(ramp(DIV, t));
}

/** value in [min,max] → sequential color. */
export function sequential(v: number, min: number, max: number): string {
  const t = max > min ? (v - min) / (max - min) : 0;
  return css(ramp(SEQ, t));
}

/** Readable text color for a given `rgb(r,g,b)` cell fill: light on dark cells,
 * dark on light cells (perceptual luminance). Works for any colormap. */
export function textColor(bg: string): string {
  const m = bg.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (!m) return '#1a1207';
  const lum = 0.299 * +m[1] + 0.587 * +m[2] + 0.114 * +m[3];
  return lum < 140 ? '#fbf3df' : '#1a1207';
}
