/**
 * textViz.ts — one source of truth for the text-side visuals, so a value maps to
 * the same colour and the same cell size in every step. The embedding table, the
 * looked-up word vectors, the mixed vectors and the pooled vector are all the same
 * kind of quantity (range ≈ [-3, 3]); they share ONE magma scale and ONE column
 * geometry, so a column pulled out of the table is pixel-identical to its source.
 */
import type { ClapBundle } from '../data/clapData';

// shared column geometry for every 64-d feature column (table + vectors + pooled)
export const COL_W = 22; // column width, px
export const CELL_H = 6; // height of one of the 64 cells, px
export const COL_GAP = 7; // gap between columns

// shared cell size for the 512-d embedding blocks (audio + text), so they match
export const EMB_CELL = 10;
export const EMB_COLS = 32;

/** Fixed magma scale for the feature layers — one range across the whole text side. */
export function featureScale(bundle: ClapBundle): { lo: number; hi: number } {
  let lo = Infinity, hi = -Infinity;
  const eat = (arr: number[]) => {
    for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
  };
  eat(bundle.weights.embTable.data);
  for (const it of bundle.text) {
    eat(it.rawEmbed.data);
    eat(it.tokenEmbed.data);
    eat(it.latent);
  }
  return { lo, hi };
}

/** Symmetric magma scale shared by both 512-d embeddings (audio + text). */
export function embedScale(bundle: ClapBundle): { lo: number; hi: number } {
  let m = 1e-6;
  for (const it of bundle.text) for (const v of it.textEmbed) m = Math.max(m, Math.abs(v));
  return { lo: -m, hi: m };
}
