# CLAP Explainer & Toy Model

An interactive explainer and a working toy implementation of **CLAP (Contrastive Language-Audio Pretraining)** — CLIP's recipe applied to sound. A recording and a phrase both become a point in one 512-dimensional space, and "similar" means "points the same way." Live at [adamsohn.com](https://adamsohn.com).

## 🚀 Project Overview

The repo has two parts.

### 1. `post/` — the interactive explainer

A visual, step-by-step article built with **Svelte 5 + Vite**. It follows one engine recording all the way to its 512-d embedding, one operation at a time, then does the same for words and measures the match. **Every audio value is computed live in your browser** from the actual sound — the in-browser forward pass is verified bit-exact against the PyTorch reference (cosine 1.0000), so nothing is a mock-up.

**What it covers:**
- Waveform → mel spectrogram, with the Fourier transform made visible
- The convolution hierarchy (find patterns, then shrink), worked out cell by cell
- Global average pool → 64, and the 64 → 512 projection
- The text side: words → token ids → embedding-table lookup → pool → 512
- The match: cosine similarity as a single dot product, plus the full contrastive matrix
- An opening nebula of **2,600 real LAION-CLAP embeddings** of car-fault sounds, on the unit sphere

**Run it locally:**
```bash
cd post
npm install
npm run dev
```

### 2. `toy/` — the miniature model + capture

Python that trains a small CLAP end to end and exports the numbers the explainer draws.

**Key files:**
- `clap_toy.py` — a miniature CLAP (audio encoder → 4 conv stages → pooled latent; text encoder → tokens → pool; two 64 → 512 projection heads; contrastive training). It captures weights and activations to `post/public/data/`.
- `hero_field.py` — builds the opening sphere from **real** LAION-CLAP embeddings (PCA → 3-D, axis-balanced onto the unit sphere, k-means colour groups, named-fault centroids).

**Run the toy:**
```bash
# audio + text side (regenerates weights.json, clips, text, similarity)
.venv/bin/python toy/clap_toy.py

# the opening nebula (needs a LAION-CLAP embeddings .npz; see the path in the script)
.venv/bin/python toy/hero_field.py
```

## 🧠 What is CLAP?

CLAP trains an audio encoder and a text encoder together so a sound and its description land next to each other in a shared embedding space. Because both sides produce the same kind of object — an L2-normalised 512-d vector — you can compare them with a single dot product (cosine similarity). That buys zero-shot audio classification, text-to-audio and audio-to-text search, and label-free anomaly flagging, with no classifier trained for the task. It is the same "encode each modality into one space, compare by angle" machinery behind CLIP for images and the JEPA family for video. The real model this mirrors is [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP).

---

*Educational project: a sandbox for understanding joint audio-text embeddings, one operation at a time. The toy is deliberately tiny (mel 32×32, conv channels 8→16→32→64, projected to CLIP's 512) so every intermediate tensor fits on screen.*
