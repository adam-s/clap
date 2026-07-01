# CLAP post — build plan

A single-page explainer, in the style of the GRPO post, for how CLAP turns
audio and text into 512-d embeddings in a shared space. Built in
`~/Projects/clap` (Svelte 5 + Vite + d3/`<canvas>`). The CLAP source lives at
`~/Projects/Temp/CLAP`; the GRPO post (`~/Projects/grpo`) and the audio-cleaning
post (`~/Projects/separate`) are the two references.

Decisions locked:
- **Toy CLAP, not the real checkpoint.** The page must run in a browser and on
  mobile, so data ships as GRPO-style captured JSON and stays tiny. Exhaustive
  per-stage capture of the real 1.8 GB HTSAT is far too large for that, so we
  build a *miniature* dual encoder — small enough that every value (mel, every
  stage activation, the 512 vector, the similarity matrix) is drawable
  cell-for-cell and ships at GRPO scale (single-digit MB). This is exactly the
  GRPO move: a toy sized to be visualized in full.
- **Keep the real story shape**: audio → mel "image" → hierarchical encoder →
  **512** embedding; text → tiny encoder → **512**; cosine-similarity diagonal.
  The 512 output dim is kept (512 cells is cheap to draw and ships fine); the
  *toy-ness* is in the encoder depth/width and the mel resolution.
- **Reuse the car-diagnosis audio from `~/Projects/separate`** (knock, exhaust,
  wheel-bearing, etc.) as the audio↔caption pairs — thematic + visual
  consistency across the two audio posts, and a natural contrastive demo
  (engine sound ↔ diagnosis text).
- Real CLAP (`630k-audioset-best.pt`, HTSAT-base) remains the *reference* the
  toy is modeled on — see the pipeline table below — but is not shipped or run
  in the browser.

## The "pixel-perfect" rule

Every figure is a real tensor from a real forward pass of the **toy** model,
drawn value-for-value, with on-screen dimensions equal to the tensor's
dimensions — the GRPO sourcing rule. The browser draws; it never runs a model.
We capture once in Python (in `~/Projects/clap`, CPU torch, no giant
checkpoint), dump every intermediate to JSON, and replay. The real CLAP shapes
below are the blueprint the toy mirrors at smaller scale.

## The real pipeline (what we capture)

### Audio tower — HTSAT-base (non-fusion)

Waveform @ 48 kHz → mel filterbank → spectrogram "image" → Swin hierarchy →
pooled latent → projection → L2-normalized 512.

Mel params, fixed in [model.py:403-408](../../Temp/CLAP/src/laion_clap/clap_module/model.py#L403-L408):
48 kHz, **64 mel bins**, `n_fft`/`window` 1024, `hop` 1024, `fmin` 50,
`fmax` 14000. HTSAT reshapes the `64×T` mel to a `256×256` image
(`freq_ratio = spec_size // mel_bins = 256/64 = 4`).

| Stage | Tensor shape (tokens × dim) | Notes |
|---|---|---|
| Waveform | `1 × ~480000` | 10 s @ 48 kHz |
| Mel spectrogram | `64 × T` | the "image"; 64 mel bins tall |
| Reshaped image | `256 × 256 × 1` | HTSAT input |
| PatchEmbed (Conv2d, stride 4) | `64 × 64 × 96` | grid of patch tokens |
| Swin **stage 1** (2 blocks) → merge | `64×64×96` → `32×32×192` | window attn + shifted-window |
| Swin **stage 2** (2 blocks) → merge | `32×32×192` → `16×16×384` | |
| Swin **stage 3** (6 blocks) → merge | `16×16×384` → `8×8×768` | deepest stack |
| Swin **stage 4** (2 blocks) | `8×8×768` | `num_features = 96·2³ = 768` |
| Norm + pool | `768` | token-semantic head / avgpool |
| Audio projection (`768→512→512`, ReLU) | `512` | |
| L2 normalize | `512` | the audio embedding `E^a` |

Config: `embed_dim=96`, `depths=[2,2,6,2]`, `num_heads=[4,8,16,32]`,
`window_size=8`, `patch_stride=4`
([htsat.py:624-627](../../Temp/CLAP/src/laion_clap/clap_module/htsat.py#L624-L627)).

### Text tower — RoBERTa-base

Sentence (labels first augmented keyword→caption) → tokenizer → RoBERTa
(12 layers, width **768**) → pooled CLS → `text_projection` (`768→512→512`,
ReLU, [model.py:500-503](../../Temp/CLAP/src/laion_clap/clap_module/model.py#L500-L503))
→ L2 normalize → 512 text embedding `E^t`.

### Joint space

`joint_embed_shape = 512` ([model.py:431](../../Temp/CLAP/src/laion_clap/clap_module/model.py#L431)).
Capture a set of N audio clips × N captions, compute the cosine-similarity
matrix `E^a_i · E^t_j` (the bright diagonal in the paper figure), scaled by the
learned `logit_scale`.

## Capture script (Python, offline)

`capture/capture_clap.py` (new), run once:

1. `pip install laion-clap`; `model.load_ckpt()` (downloads
   `630k-audioset-best.pt`, ~1.8 GB).
2. Register forward hooks on: mel transform output, `patch_embed`, each
   `BasicLayer`/`PatchMerging`, final norm/pool, audio projection; and on the
   RoBERTa layers + text projection.
3. Run ~6–8 curated clips (reuse `separate`'s clips where licensing allows:
   exhaust, knock, music, talk, etc.) + their captions.
4. Dump per-clip JSON: waveform (downsampled for display), `64×T` mel, the
   `256×256` reshaped image, every stage activation map (mean over channels +
   a few representative channels, to bound size), pooled latent, both 512
   vectors. Dump a `clips.json` manifest and a `similarity.json` matrix.
5. Quantize/trim so each file stays in the low-MB range (GRPO's
   `trajectory.json` is 1.9 MB; weight series 6.9 MB — comparable budget).

Open question to resolve at build time: full `H×W×C` activation maps are huge
(`64·64·96 ≈ 393k` floats for stage 1 alone). Plan: store the per-token
channel-mean map at full `H×W` (real, pixel-perfect spatially) plus a handful
of named channels, not all 96/192/384/768. Document the truncation in-page
(GRPO's "no silent caps" rule).

## Page structure (sections)

Mirrors the paper figure, left tower then right tower then the join:

0. **Opening** — hook: drop a sound in, watch it become 512 numbers that line
   up with words.
1. **Two towers, one space** — the CLAP overview (audio encoder ↑, text
   encoder ↑, shared 512-d space, contrastive diagonal).
2. **Waveform** — the raw 48 kHz signal for the chosen clip.
3. **Audio → image** — the mel filterbank: 64 bins, real params; the waveform
   becomes the `64×T` spectrogram. The "converted to an image" beat.
4. **Into HTSAT** — reshape to `256×256`, PatchEmbed → `64×64×96` token grid.
5. **The Swin hierarchy** — the exhaustive part: scrub through 4 stages,
   activation maps shrinking `64→32→16→8` while channels grow
   `96→192→384→768`. GRPO-style real-value cell rendering.
6. **Collapse to 512** — pool + projection → the audio embedding strip (512
   real cells, L2-normalized).
7. **The text tower** — caption → tokens → RoBERTa → 512 text strip.
8. **The shared space** — cosine-similarity matrix with its diagonal; the
   contrastive objective in one figure. Optionally a 2-D projection of the
   joint space (clearly labeled as a projection, not pixel-perfect).
9. **Conclusion / play** — pick any clip + any caption, watch both land and
   the matching score light up.

The centerpiece animation (sections 2–6): one continuous scrub where the
waveform morphs into the mel image, the image patchifies, the maps shrink
through the hierarchy, and the whole thing collapses into the 512 strip — data
conserved at every frame.

## Frontend architecture (mirror GRPO)

- **Cursor store** — but the cursor is **pipeline stage**, not training step.
  CLAP inference is one forward pass, so the playhead walks *stages* (waveform
  → mel → patch → stage1…4 → 512), not time. Plus a **clip selector** store
  (which captured example) and a **caption selector** for the text side.
- **Loaders** — `clipData.ts` (fetch per-clip JSON, cache singleton, like
  `trajectoryData.ts`), `similarityData.ts`.
- **Viz components** (`<canvas>`/d3):
  - `Waveform.svelte`
  - `MelSpectrogram.svelte` — possibly adapt `separate`'s existing one.
  - `ReshapeToImage.svelte` — the `64×T` → `256×256` morph.
  - `PatchGrid.svelte` — the `64×64` patch tokens.
  - `SwinStageMaps.svelte` — the exhaustive per-stage `H×W` activation
    heatmaps; GRPO-cell renderer reused/adapted.
  - `VectorStrip.svelte` — a 512-cell embedding strip (audio + text).
  - `SimilarityMatrix.svelte` — the cosine matrix + diagonal.
- **Playback bar** — stage scrubber + play, like GRPO's `PlaybackBar`.

## Build order

1. Scaffold the Svelte+Vite app in `~/Projects/clap` (copy GRPO's config,
   `base: './'`, Layout/Section/Prose/PlaybackBar skeleton).
2. Write `capture/capture_clap.py`, run it on 1 clip, lock the JSON schema.
3. Build sections 2–6 (the audio centerpiece) against that one clip.
4. Add the text tower + similarity (sections 7–8) and the clip/caption
   pickers.
5. Opening + conclusion + interactive play.
6. Build + deploy alongside the other posts.

## Open questions / dependencies

- **Python env**: is there a working env to `pip install laion-clap` and
  download the 1.8 GB checkpoint? This is the first hard dependency.
- **Clips + captions**: reuse `separate`'s audio, or pick a fresh set with
  clean licensing? Need the audio files to capture from.
- **Activation-map budget**: confirm the channel-mean + few-channels plan vs.
  full `H×W×C` (size blowup).
