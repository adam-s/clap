# Reference — how the GRPO post replays a model's training values

Reference notes on a **separate** project we're drawing from: the GRPO
explainer at `~/Projects/grpo` (published at `adamsohn.com/grpo/`). It is a
single-page article that replays a real RL training run — the model is
trained once, offline; every training step is dumped to JSON; the page fetches
the JSON and re-tints every embedded figure as you drag one playhead.

This is the "Transformer-Explainer" pattern (one recorded trajectory, scrubbed
in the browser) applied to a GRPO training loop instead of a single inference
pass. We're documenting it here because we want the same mechanism in
`~/Projects/clap`.

All file links below point into the sibling `~/Projects/grpo` repo (and
`~/Projects/blog` for the deployed build), assuming both sit next to `clap`
under `~/Projects`.

---

## 1. The one-sentence model

> A toy transformer is trained once with GRPO; every training step is logged
> to JSON; the page fetches that JSON and re-colors every figure as you scrub
> a single training-step playhead.

Everything below expands those three clauses: where the JSON comes from,
what's in it, how it's loaded, and how scrubbing wires through to pixels.

**What it is *not*:**

- **Not live training.** No gradients run in the browser. The trajectory is
  fixed at build time; the article is a deterministic replay.
- **Not a generic dashboard.** Not TensorBoard. The figures are hand-built
  explanatory visuals (token-level ratio/KL/advantage matrices, a 3D
  weight-stack heatmap, a Rubik's-cube solver animation), each bound to the
  same step cursor.

## 2. Two repos, one artifact

| Location | Role | Contents |
|---|---|---|
| `~/Projects/grpo/` | **Source** | [post/](../../grpo/post/) — the Svelte 5 + Vite app. [toy/](../../grpo/toy/) — the Python toy model that generates the data. |
| `~/Projects/blog/grpo/` | **Build output** | The `vite build` result, copied into the blog repo. Served at `adamsohn.com/grpo/`. Just `index.html` + hashed JS/CSS + the `toy/` and `data/` JSON assets. |

You edit `post/`, build it, and copy `dist/` to `blog/grpo/`.

### The single page

[post/src/main.ts](../../grpo/post/src/main.ts) mounts
[App.svelte](../../grpo/post/src/App.svelte) into `#app`.
[App.svelte:20-33](../../grpo/post/src/App.svelte#L20-L33) stacks twelve prose
sections inside a `<Layout>`:

```
Opening → Background → Models → GroupSampling → Reward → Advantage →
Ratio → Clip → KL → PessimisticMin → Objective → Conclusion
```

Each section explains one term of the GRPO objective and embeds a figure
driven by the recorded run. The whole article is one scroll; there is no
client routing for the reader. `main.ts` does carry a small hash router, but
only for standalone debug views (`#/pipeline/flow`, `#/pipeline/grid`,
`#/pipeline/model`, `#/pipeline/grpo`) — development scaffolds, not part of the
published experience.

## 3. Where the numbers come from — the toy run

The training values shown in the article all come from one offline run of a
toy transformer.

### The model

A tiny decoder-only transformer ([toy/model.py](../../grpo/toy/model.py)). Its
architecture is frozen in
[post/public/toy/config.json](../../grpo/post/public/toy/config.json):

| Field | Value |
|---|---|
| `vocab_size` | 48 |
| `d_model` | 64 |
| `n_layers` | 2 |
| `n_heads` | 4 |
| `d_ff` | 128 |
| `max_seq_len` | 256 |

The 48-token vocabulary is purpose-built for one task: solving scrambled
Rubik's cubes. It holds the move tokens (`U`, `U'`, `U2`, `D`, … `B2`), the
color/face tokens (`COL_W`, …), structural tags (`<thinking>`, `<answer>`,
…), and a few English words. The model reads a scramble, "thinks," and emits a
move sequence.

The pipeline is three scripts:

- [toy/gen_corpus.py](../../grpo/toy/gen_corpus.py) — generates
  scramble/solution data.
- [toy/sft.py](../../grpo/toy/sft.py) — supervised fine-tuning for a sane
  starting policy. The SFT checkpoint becomes the frozen reference policy
  `π_ref`.
- [toy/train_grpo.py](../../grpo/toy/train_grpo.py) — the GRPO loop that
  produces the trajectory the article replays.

### The GRPO loop

[train_grpo.py:190](../../grpo/toy/train_grpo.py#L190) (`train`) loads the SFT
checkpoint as both the initial policy and the frozen reference. Each step, for
one scramble prompt, it:

1. **Samples a group** of `G` rollouts (default `G=8`) from the current
   policy.
2. **Scores** each rollout with a rule-based reward (cube progress, format
   tags, move-parse validity, solved bonus, brevity).
3. **Computes group-relative advantages** — standardize the `G` rewards within
   the group (mean as baseline, std as scale). No critic network.
4. **Runs PPO-style inner epochs** (default 4): per-token new/old/ref
   log-probs, clipped ratios, the pessimistic min, a KL penalty to `π_ref`,
   and a gradient step.

Relevant defaults from `parse_args()`
([train_grpo.py:500](../../grpo/toy/train_grpo.py#L500)):

| Arg | Default | Meaning |
|---|---|---|
| `--G` | 8 | rollouts per group |
| `--beta-kl` | 0.04 | KL penalty weight (β) |
| `--clip-eps` | 0.2 | PPO clip ε |
| `--ppo-epochs` | 4 | inner optimization epochs per step |
| `--lr` | 1e-4 | learning rate |
| `--log-detail-every` | 10 | record full per-token detail every N steps |
| `--snapshot-every` | 50 | weight-snapshot cadence |
| `--k-max` | 5 | curriculum: max scramble depth |

The committed run is 500 steps with a curriculum that ramps scramble
difficulty (`k`) as the policy's success rate climbs.

### What each step records

The loop writes one JSON record per step, keyed by `tier`:

- **`detail`** steps (every `log_detail_every`, plus the last) carry the full
  per-token `completions` array — this drives the in-section token matrices.
- **`summary`** steps carry only scalars (rewards, advantages, group stats) —
  enough for the training curves, not the per-token heatmaps.

**Step-level fields** (`RawStep`, typed at
[trajectoryData.ts:31](../../grpo/post/src/lib/viz/trajectoryData.ts#L31)). The
on-disk record also includes `k`, `k_configured`, `success_ema`, `bumped`,
`step_stats`, `key_position_probs`, `prompt_tokens`.

| Field | Type | What it is |
|---|---|---|
| `step` | int | training step index (0–499) |
| `scramble` | string | the cube scramble for this prompt |
| `tier` | `'detail' \| 'summary'` | detail = has `completions` |
| `rewards` | `number[G]` | raw reward per rollout |
| `advantages` | `number[G]` | group-standardized advantage per rollout |
| `reward_components` | `{format_tags, moves_parse, progress, solved, brevity}[G]` | reward breakdown |
| `group_stats` | `{reward_mean, reward_std, clip_fraction?, kl_to_ref?}` | aggregate stats |
| `rollout_previews` | `{text_preview, n_tokens, solved}[G]` | short text + solved flag (kept on summary steps too) |
| `completions` | `RawCompletion[G]` | **detail only** — per-token arrays (below) |
| `probe` | `{scramble, tokens, logprobs_curr, logprobs_prev, delta}` | fixed probe re-scored each detail step to show what the gradient step moved |

**Per-token fields** (`RawCompletion`, one per rollout on a detail step,
typed at
[trajectoryData.ts:11](../../grpo/post/src/lib/viz/trajectoryData.ts#L11)):

| Field | Type | What it is |
|---|---|---|
| `text` | string | decoded completion |
| `token_ids` / `token_names` | `number[T]` / `string[T]` | generated tokens + labels |
| `new_logprobs` | `number[T]` | log π_θ (current policy) per token |
| `ref_logprobs` | `number[T]` | log π_ref (frozen SFT) per token |
| `ratios` | `number[T]` | π_θ / π_old per token |
| `clipped` | `number[T]` | clipped ratio per token |
| `kl_per_tok` | `number[T]` | KL(π_θ‖π_ref) per token |
| `mask` | `number[T]` | 1 for real tokens, 0 for padding |
| `first_tok_new_probs?` / `first_tok_ref_probs?` | `number[vocab]` | full softmax at t=0 under each policy |

The `mask` matters: matrix extractors keep only masked-in positions, so
padding never renders as a colored cell.

## 4. The data files

What ships in the build, and whether the current article actually reads it.

### `public/toy/` — the toy run (live)

| File | Size | Status | Consumer |
|---|---|---|---|
| `trajectory.json` | 1.9 MB | **live** | `loadTrajectory()` |
| `weights_series.json` | 6.9 MB | **live** | `loadWeightsSeries()` |
| `weights.json` | 631 KB | **fallback** | `loadWeightsSeries()` if the series 404s |
| `config.json` | 482 B | reference | architecture constants |
| `activations.json` | 1.0 MB | present, not fetched | single-snapshot activations |
| `activations_trajectory.json` | 6.6 MB | present, not fetched | per-step activations |
| `eval.json` | 2.4 KB | present, not fetched | held-out eval samples |

`trajectory.json` is the spine — a JSON **array of 500 step objects**; 51 are
`tier: "detail"`, 449 are `tier: "summary"`.

`weights_series.json` is a sparse map of per-step weight snapshots:

```jsonc
{
  "schema": 1,
  "steps": [0, 50, 100, ...],            // snapshot step numbers
  "snapshots": {
    "0":  { "<tensor>": { "shape": [r, c], "data": [...] }, ... },
    "50": { ... }
  }
}
```

The weight heatmaps interpolate between adjacent snapshots and render the
*delta from the first snapshot*, so cells that never move stay at true zero.

`activations*.json` and `eval.json` are produced by the toy capture step and
shipped, but the current article has no `fetch` for them — available but
dormant.

### `public/data/` — auxiliary + legacy

| File | Status | Consumer |
|---|---|---|
| `trace-moves.json` (103 KB) | **live** | [CubeGridSimple.svelte:146](../../grpo/post/src/lib/rubiks/CubeGridSimple.svelte#L146) |
| `meta.json` (608 B) | reference | run metadata (below) |
| `anchor_step.json` (57 KB) | legacy, not fetched | — |
| `kl_compare.json` (98 KB) | legacy, not fetched | — |
| `trajectory_summary.json` (1.9 MB) | legacy, not fetched | — |

Only `trace-moves.json` is wired in: the Rubik's-cube animation fetches it,
filters records by scramble depth, and animates each solve. The other three
are **not referenced anywhere in `src/`** (verified by grep) — leftovers from a
different, larger run. `meta.json` confirms it describes a real
**SmolLM2-360M-Instruct** GRPO run (`run_id`
`2026-04-20T07-56-35_curriculum_k1_to_k20`, 4000 steps, lr 5e-6), not the toy
model.

So when tracing why an on-screen number doesn't match a file: the token
matrices and weight heatmaps come from `public/toy/`, the cube animation from
`public/data/trace-moves.json`, and the SmolLM `public/data/*` files feed
nothing in the current build.

## 5. Loaders & stores

Three loader modules turn raw JSON into typed view data; two Svelte stores
turn the current step into per-figure props.

```
trajectory.json ──► trajectoryData.ts ──► pipelineData.ts ──┐
                          │                                  ├─► grpo-flow.ts (grpoData)
                    grpo.ts (featuredStep, detailSteps) ─────┘
weights_series.json ─► weightsSeries.ts ──────────────────────► ModelDiagram.svelte
```

### `trajectoryData.ts` — raw loader + matrix extractors

[post/src/lib/viz/trajectoryData.ts](../../grpo/post/src/lib/viz/trajectoryData.ts)

- **`loadTrajectory()`** ([:65](../../grpo/post/src/lib/viz/trajectoryData.ts#L65))
  — fetches `${BASE_URL}toy/trajectory.json` once and caches the array in a
  module-level singleton.
- **`detailStepNumbers(traj)`** ([:81](../../grpo/post/src/lib/viz/trajectoryData.ts#L81))
  — the step numbers with `completions`. This is the set the playhead walks
  (51 steps, not all 500).
- **`extractGxT(step, field)`** ([:98](../../grpo/post/src/lib/viz/trajectoryData.ts#L98))
  — the workhorse. Builds a **G×T matrix** for a per-token field: rows =
  rollouts (G), columns = token positions (T = longest masked rollout), only
  `mask > 0` positions kept, shorter rows right-padded with `NaN`. Returns
  `{ matrix, tokenNames, rowMeta }`, where `rowMeta[g]` carries that rollout's
  `reward`, `advantage`, and `solved` flag. These matrices are what every
  token-level heatmap renders.
- **`buildStepSummaries(traj)`** ([:166](../../grpo/post/src/lib/viz/trajectoryData.ts#L166))
  — collapses every step to scalars (`rewardMean`, `solvedFraction`,
  `clipFraction`, `klToRef`, …) for the curves spanning all 500 steps.

### `pipelineData.ts` — one step → all derived matrices

[post/src/lib/viz/pipelineData.ts](../../grpo/post/src/lib/viz/pipelineData.ts).
**`buildPipelineData(step)`**
([:56](../../grpo/post/src/lib/viz/pipelineData.ts#L56)) takes one `RawStep`
and assembles the full `PipelineStepData` the figures need. It calls
`extractGxT` per field, then derives the two fields the trajectory doesn't
store: **`pessimisticMin[g][t]`** = `min(ratio·A, clip(ratio)·A)` and
**`objectivePerTok[g][t]`** (including the `BETA_KL` = 0.04 penalty). It also
unpacks the first-token softmax distributions and copies through
`rewardComponents`, `groupStats`, `solved`, `tokenNames`, `probe`.

### `weightsSeries.ts` — animatable weight snapshots

[post/src/lib/viz/weightsSeries.ts](../../grpo/post/src/lib/viz/weightsSeries.ts)

- **`loadWeightsSeries(base)`** ([:38](../../grpo/post/src/lib/viz/weightsSeries.ts#L38))
  — fetches `weights_series.json`; on failure falls back to single-snapshot
  `weights.json`. Precomputes per-tensor max|Δ| across the run so diverging
  colormaps use a fixed scale.
- **`getInterpolatedDeltaAt(key, step)`** ([:108](../../grpo/post/src/lib/viz/weightsSeries.ts#L108))
  — the one the heatmap uses. Linearly interpolates each cell between the two
  bracketing snapshots, returns the **delta from the first snapshot** plus the
  tensor's `absMax`. Delta-from-init means every tick moves color (no
  step-jump at snapshot boundaries) and unmoved weights render at true zero —
  so the viewer sees what the policy *learned*, not the pattern it was born
  with. Output buffers are reused per tensor to avoid per-frame allocation.

### `grpo.ts` — the step cursor and playback clock

[post/src/lib/stores/grpo.ts](../../grpo/post/src/lib/stores/grpo.ts)

- **`featuredStep`** ([:7](../../grpo/post/src/lib/stores/grpo.ts#L7)) —
  `writable<number>`, the single source of truth for "where in training are
  we." Initial value 5.
- **`detailSteps`** ([:24](../../grpo/post/src/lib/stores/grpo.ts#L24)) —
  read-only, populated from `loadTrajectory().then(detailStepNumbers)`. The
  list the playhead may land on.
- **`pipelinePlaying`** ([:14](../../grpo/post/src/lib/stores/grpo.ts#L14)) —
  boolean. When true, a `setInterval` loop
  ([:33](../../grpo/post/src/lib/stores/grpo.ts#L33)) advances `featuredStep`
  to the next detail step every `PLAY_INTERVAL_MS` (900 ms), stopping at the
  end.

### `grpo-flow.ts` — the derived per-step view store

[post/src/lib/stores/grpo-flow.ts](../../grpo/post/src/lib/stores/grpo-flow.ts).
Modeled on CNN-Explainer's `cnnStore`: one store every figure subscribes to.

- **`grpoData`** ([:32](../../grpo/post/src/lib/stores/grpo-flow.ts#L32)) —
  `derived([featuredStep, _trajReady])`. Resolves the current step against the
  cached trajectory; if that exact step has no `completions`, snaps to the
  **nearest detail step**, then returns `buildPipelineData(step)`.
- **`focus` / `hovered`** ([:52](../../grpo/post/src/lib/stores/grpo-flow.ts#L52))
  — the `{g, t}` cell the reader is inspecting; re-clamped when `grpoData`
  reloads, defaulting `g` to the first solved rollout.
- **`focusedSummary`** ([:84](../../grpo/post/src/lib/stores/grpo-flow.ts#L84))
  — `derived([grpoData, focus])`, collapses the focused cell to flat scalars
  (token, reward, advantage, log-probs, ratio, clipped, pmin, KL, J, loss,
  solved) for the inspector.
- **`clipEps` / `betaKL`** ([:73](../../grpo/post/src/lib/stores/grpo-flow.ts#L73))
  — slider knobs, **display-only on the current pass**: matrices are passed
  through from recorded data as-is, not re-derived live under slider values.

The split is deliberate: `grpo.ts` owns *time* (which step, playing or not);
`grpo-flow.ts` owns *the data at that time*.

## 6. Playback & binding — one scrub, whole page

[PlaybackBar.svelte](../../grpo/post/src/lib/components/PlaybackBar.svelte) is a
control pinned to the bottom, rendered once by `App.svelte`. It owns no data —
it reads/writes the `grpo.ts` stores:

| Control | Effect |
|---|---|
| Play / pause | toggles `pipelinePlaying`; the 900 ms loop walks detail steps |
| Next / prev | sets `featuredStep` to the next/previous detail step |
| Reset | stops playback, sets `featuredStep` to the first detail step |
| Scrub | snaps to the nearest detail step |

Keyboard (bound on `window`, ignored while typing): **space** = play/pause,
**→** = next, **←** = prev. The bar walks the **detail** steps (51), so every
tick lands on a step with full per-token completions — no dead frames.

### The binding chain

```
 user drags PlaybackBar
        │  writes
        ▼
   featuredStep  (grpo.ts)
        │
        ├──────────────► grpoData  (grpo-flow.ts, derived)
        │                   │  buildPipelineData(nearest detail step)
        │                   ▼
        │            every flow/grid column + in-section figure
        │            (rewards, advantages, ratios, clipped, KL,
        │             pmin, objective, softmax bars, inspector)
        │                   │ retint
        │                   ▼
        │                 pixels
        │
        └──────────────► ModelDiagram.svelte
                            │  getInterpolatedDeltaAt(key, featuredStep)
                            ▼
                       3D weight-stack heatmap retints
```

Two consumer families, both keyed on the same `featuredStep`:

1. **Token-level figures** subscribe to `grpoData`. When `featuredStep`
   changes, `grpoData` recomputes (snapping to the nearest detail step) and
   every subscribing column gets the new step's matrices.
2. **The weight heatmap**
   ([ModelDiagram.svelte](../../grpo/post/src/lib/components/viz/ModelDiagram.svelte))
   subscribes to `featuredStep` directly and calls
   `getInterpolatedDeltaAt(tensor, featuredStep)` per tensor, animating
   smoothly between sparse snapshots. (Its 2D-canvas 3D rendering is its own
   topic, documented in
   [grpo's docs/3d-on-2d/](../../grpo/docs/3d-on-2d/README.md).)

Because both families read the same store, the reward matrix, the KL matrix,
the per-token ratios, and the weight deltas all show the **same training step**
at every instant. Scrubbing is a synchronized replay, not independent widgets.

### How a section embeds a figure

The prose sections (e.g.
[04_Advantage.svelte](../../grpo/post/src/sections/04_Advantage.svelte)) are
mostly KaTeX math and copy. They embed data-driven visuals as components
(`OpGrid`, `FocusDemo`, flow columns) which subscribe to `grpoData`. The
section file itself never touches the trajectory — it drops in the figure and
lets the store wire it up. Adding a new step-aware figure is cheap: subscribe
to `grpoData` (or `featuredStep`), render, done — playback already drives it.

## 7. Build & deploy

```bash
cd post
npm install
npm run dev      # vite, port 5174
```

[vite.config.ts](../../grpo/post/vite.config.ts) sets **`base: './'`** and
`server.port: 5174`. The dev `index.html` loads `/src/main.ts`; Vite serves
`public/` at the root, so `fetch('toy/trajectory.json')` resolves against the
dev server.

`base: './'` makes every asset path relative, so the build works under any
subpath without rewrites — that's why it can live at `adamsohn.com/grpo/`
without an absolute base. In code, every fetch is written as
`` `${import.meta.env.BASE_URL}toy/trajectory.json` `` — `BASE_URL` is `'/'` in
dev and `'./'` in the build, so the same line works in both.

`npm run build` emits `post/dist/` (`index.html` referencing hashed bundles,
`assets/`, plus `toy/`, `data/`, `audio/`, `favicon.svg` copied from
`public/`). That output is copied into `~/Projects/blog/grpo/`, published as
`adamsohn.com/grpo/`. Tell source from build by `index.html`: the source one
([post/index.html](../../grpo/post/index.html)) loads `/src/main.ts`; the
deployed [blog/grpo/index.html](../../blog/grpo/index.html) loads hashed
`./assets/…`.

One gap to know: [post/package.json](../../grpo/post/package.json) wires a
`prepare-data` script to `scripts/prepare.py`, but that file is **not currently
in [post/scripts/](../../grpo/post/scripts/)**. The committed
`public/toy/*.json` were produced from a toy run, but the exact gz-JSONL → JSON
conversion isn't checked in. To regenerate the data you re-run
`train_grpo.py`, export weights with
[export_weights.py](../../grpo/toy/export_weights.py) and activations with
[capture.py](../../grpo/toy/capture.py), convert the gzipped JSONL trajectory
to a plain JSON array, copy everything into
[post/public/toy/](../../grpo/post/public/toy/), then `npm run build` and
redeploy.

## 8. Implementation map

| Concept | File / symbol | Anchor |
|---|---|---|
| App mount + hash router | `main.ts` `route()` | [main.ts](../../grpo/post/src/main.ts) |
| Section order | `App.svelte` `<Layout>` | [App.svelte:20-33](../../grpo/post/src/App.svelte#L20-L33) |
| **Step cursor (source of truth)** | `featuredStep` | [grpo.ts:7](../../grpo/post/src/lib/stores/grpo.ts#L7) |
| Allowed step set | `detailSteps` | [grpo.ts:24](../../grpo/post/src/lib/stores/grpo.ts#L24) |
| Playback clock (900 ms) | `pipelinePlaying` subscribe | [grpo.ts:31-44](../../grpo/post/src/lib/stores/grpo.ts#L31-L44) |
| Playhead UI + keys | `PlaybackBar.svelte` | [PlaybackBar.svelte](../../grpo/post/src/lib/components/PlaybackBar.svelte) |
| Trajectory fetch (cached) | `loadTrajectory()` | [trajectoryData.ts:65](../../grpo/post/src/lib/viz/trajectoryData.ts#L65) |
| Detail-step list | `detailStepNumbers()` | [trajectoryData.ts:81](../../grpo/post/src/lib/viz/trajectoryData.ts#L81) |
| G×T matrix extractor | `extractGxT()` | [trajectoryData.ts:98](../../grpo/post/src/lib/viz/trajectoryData.ts#L98) |
| All-steps scalar curves | `buildStepSummaries()` | [trajectoryData.ts:166](../../grpo/post/src/lib/viz/trajectoryData.ts#L166) |
| One step → all matrices | `buildPipelineData()` | [pipelineData.ts:56](../../grpo/post/src/lib/viz/pipelineData.ts#L56) |
| **Per-step view store** | `grpoData` (derived) | [grpo-flow.ts:32](../../grpo/post/src/lib/stores/grpo-flow.ts#L32) |
| Focused-cell scalars | `focusedSummary` | [grpo-flow.ts:84](../../grpo/post/src/lib/stores/grpo-flow.ts#L84) |
| Display-only knobs | `clipEps`, `betaKL` | [grpo-flow.ts:73-74](../../grpo/post/src/lib/stores/grpo-flow.ts#L73-L74) |
| Weight snapshots loader | `loadWeightsSeries()` | [weightsSeries.ts:38](../../grpo/post/src/lib/viz/weightsSeries.ts#L38) |
| Interpolated delta-from-init | `getInterpolatedDeltaAt()` | [weightsSeries.ts:108](../../grpo/post/src/lib/viz/weightsSeries.ts#L108) |
| 3D weight heatmap | `ModelDiagram.svelte` | [ModelDiagram.svelte](../../grpo/post/src/lib/components/viz/ModelDiagram.svelte) |
| Cube-solve animation | `CubeGridSimple.svelte` + `trace-moves.json` | [CubeGridSimple.svelte:146](../../grpo/post/src/lib/rubiks/CubeGridSimple.svelte#L146) |
| GRPO training loop (data source) | `train(args)` | [train_grpo.py:190](../../grpo/toy/train_grpo.py#L190) |
| Run args / defaults | `parse_args()` | [train_grpo.py:500](../../grpo/toy/train_grpo.py#L500) |

### Offline data flow (build time)

```
gen_corpus.py ──► sft.py ──► SFT checkpoint (= π_ref)
                                   │
                                   ▼
                            train_grpo.py  (G=8, β=0.04, ε=0.2, 500 steps)
                                   │
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                     ▼
   trajectory.jsonl.gz   activations_traj.jsonl.gz   export_weights.py
              │                                          │
        (convert + copy)                          weights_series.json
              ▼                                          ▼
        public/toy/trajectory.json            public/toy/weights_series.json
```

### Runtime data flow (per scrub)

```
featuredStep ── changes ──┐
                          │
        ┌─────────────────┴───────────────────┐
        ▼                                       ▼
   grpoData (derived)                    ModelDiagram subscribe
   = buildPipelineData(                  = getInterpolatedDeltaAt(
       nearest detail step)                   key, featuredStep)
        │                                       │
        ▼                                       ▼
   flow/grid columns + section figures    weight-stack cells retint
        │
        ▼
   focus changes ─► focusedSummary ─► inspector panel
```

### Where to start reading the source

- **The cursor:** [grpo.ts](../../grpo/post/src/lib/stores/grpo.ts) →
  [PlaybackBar.svelte](../../grpo/post/src/lib/components/PlaybackBar.svelte).
- **The data shape:** the `RawStep`/`RawCompletion` types atop
  [trajectoryData.ts](../../grpo/post/src/lib/viz/trajectoryData.ts).
- **A figure's props:** `PipelineStepData` in
  [pipelineData.ts](../../grpo/post/src/lib/viz/pipelineData.ts) and the
  `grpoData` store in
  [grpo-flow.ts](../../grpo/post/src/lib/stores/grpo-flow.ts).
