"""
clap_toy.py — a miniature CLAP, sized to be drawn in full.

Mirrors the real CLAP pipeline (audio -> mel "image" -> hierarchical encoder ->
512-d embedding; text -> encoder -> 512-d embedding; contrastive cosine matrix)
but at a toy scale so every intermediate tensor ships as small JSON and animates
on a phone. Real CLAP shapes it mirrors (HTSAT-base): mel 64xT reshaped to
256x256, 4 Swin stages 64->32->16->8 with channels 96->192->384->768, projected
768->512. Here: mel 32x32, 4 conv stages 16->8->4->2 with channels 8->16->32->64,
projected 64->512.

Run:  .venv/bin/python toy/clap_toy.py
Out:  post/public/data/*.json  (clips, embeddings, activations, similarity)
"""

from __future__ import annotations
import json, math, os, re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

torch.manual_seed(0)
np.random.seed(0)

ROOT = Path(__file__).resolve().parent.parent          # ~/Projects/clap
AUDIO_DIR = Path.home() / "Projects/separate/public/audio"
OUT_DIR = ROOT / "post/public/data"

# ── The dataset: car-diagnosis sounds ↔ their plain-language captions ─────────
# Reuse the isolated `-target` diagnostic clips from the `separate` post so the
# two audio posts stay visually consistent.
PAIRS = [
    ("knock",         "knock-target.wav",         "engine knock"),
    ("exhaust",       "exhaust-target.wav",       "exhaust leak"),
    ("wheel-bearing", "wheel-bearing-target.wav", "wheel bearing noise"),
]

SR = 16000
WIN_SEC = 1.5
WIN = int(SR * WIN_SEC)
MEL_BINS = 32
IMG = 32                       # mel reshaped to IMG x IMG (CLAP reshapes to 256)
EMBED = 512                    # joint embedding dim — kept at CLAP's 512
LATENT = 64                    # encoder latent before projection (CLAP: 768)

mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=512, hop_length=256, n_mels=MEL_BINS, f_min=50, f_max=7500
)


def load_window(path: Path, center: bool = True, offset: int | None = None) -> np.ndarray:
    w, sr = sf.read(str(path))
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != SR:
        w = torchaudio.functional.resample(torch.tensor(w).float(), sr, SR).numpy()
    if len(w) < WIN:
        w = np.pad(w, (0, WIN - len(w)))
    if offset is not None:
        start = max(0, min(offset, len(w) - WIN))
    elif center:
        start = (len(w) - WIN) // 2
    else:
        start = np.random.randint(0, max(1, len(w) - WIN))
    return w[start:start + WIN].astype(np.float32)


def wav_to_mel_image(wav: np.ndarray) -> torch.Tensor:
    """waveform -> log-mel -> standardize -> square IMGxIMG image. Returns [1,IMG,IMG]."""
    m = mel_tf(torch.tensor(wav))                  # [MEL_BINS, frames]
    m = torch.log(m + 1e-6)
    m = (m - m.mean()) / (m.std() + 1e-5)
    m = F.interpolate(m[None, None], size=(IMG, IMG), mode="bilinear", align_corners=False)
    return m[0]                                     # [1, IMG, IMG]


# ── Text tokenizer (toy, word-level over the caption vocabulary) ─────────────
def build_vocab(captions: list[str]) -> dict[str, int]:
    words = ["<pad>"]
    for c in captions:
        for tok in re.findall(r"[a-z]+", c.lower()):
            if tok not in words:
                words.append(tok)
    return {w: i for i, w in enumerate(words)}


def encode_text(caption: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab[t] for t in re.findall(r"[a-z]+", caption.lower())]
    return ids + [0] * (max_len - len(ids))


# ── Models ───────────────────────────────────────────────────────────────────
class AudioEncoder(nn.Module):
    """mel image -> 4 conv stages (halving each) -> pooled latent. Mirrors HTSAT."""
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(1, 8, 3, stride=2, padding=1)   # 32 -> 16
        self.s1 = nn.Conv2d(8, 16, 3, stride=2, padding=1)     # 16 -> 8
        self.s2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)    # 8 -> 4
        self.s3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)    # 4 -> 2

    def forward(self, x, capture: dict | None = None):
        p = F.relu(self.patch(x))
        a1 = F.relu(self.s1(p))
        a2 = F.relu(self.s2(a1))
        a3 = F.relu(self.s3(a2))
        latent = a3.mean(dim=(2, 3))                           # global avgpool -> [B,64]
        if capture is not None:
            capture["patch"] = p
            capture["stage1"] = a1
            capture["stage2"] = a2
            capture["stage3"] = a3
            capture["latent"] = latent
        return latent


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, max_len: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, LATENT, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, LATENT))
        layer = nn.TransformerEncoderLayer(LATENT, nhead=4, dim_feedforward=128,
                                           batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, ids, capture: dict | None = None):
        mask = ids == 0
        raw = self.emb(ids)                                    # table lookup, pre-context
        h = self.tr(raw + self.pos[:, : ids.shape[1]], src_key_padding_mask=mask)
        keep = (~mask).float()[..., None]
        latent = (h * keep).sum(1) / keep.sum(1).clamp(min=1)  # masked mean pool
        if capture is not None:
            capture["raw"] = raw                              # [B,T,64] embedding lookup
            capture["tokens"] = h                             # [B,T,64] after context mixing
            capture["latent"] = latent                        # [B,64] pooled
        return latent


class ProjHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(LATENT, EMBED)
        self.l2 = nn.Linear(EMBED, EMBED)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


class ToyCLAP(nn.Module):
    def __init__(self, vocab_size: int, max_len: int):
        super().__init__()
        self.audio = AudioEncoder()
        self.text = TextEncoder(vocab_size, max_len)
        self.audio_proj = ProjHead()
        self.text_proj = ProjHead()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def embed_audio(self, img, capture=None):
        z = self.audio_proj(self.audio(img, capture))
        return F.normalize(z, dim=-1)

    def embed_text(self, ids, capture=None):
        z = self.text_proj(self.text(ids, capture))
        return F.normalize(z, dim=-1)


def main():
    slugs = [p[0] for p in PAIRS]
    files = [p[1] for p in PAIRS]
    captions = [p[2] for p in PAIRS]
    N = len(PAIRS)

    vocab = build_vocab(captions)
    max_len = max(len(re.findall(r"[a-z]+", c)) for c in captions)
    text_ids = torch.tensor([encode_text(c, vocab, max_len) for c in captions])

    model = ToyCLAP(len(vocab), max_len)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    paths = [AUDIO_DIR / f for f in files]
    for p in paths:
        assert p.exists(), f"missing audio: {p}"

    # ── Train: contrastive over the N pairs, fresh random audio crops each step ──
    STEPS = 400
    traj = []   # similarity-matrix snapshots over training, for the replay anim
    model.train()
    for step in range(STEPS):
        imgs = torch.stack([wav_to_mel_image(load_window(p, center=False)) for p in paths])
        za = model.embed_audio(imgs)
        zt = model.embed_text(text_ids)
        scale = model.logit_scale.exp().clamp(max=100)
        logits = scale * za @ zt.t()
        labels = torch.arange(N)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 8 == 0 or step == STEPS - 1:
            with torch.no_grad():
                sim = (za @ zt.t()).cpu().numpy()
            traj.append({"step": step, "sim": np.round(sim, 4).tolist()})
        if step % 50 == 0:
            acc = (logits.argmax(1) == labels).float().mean().item()
            print(f"step {step:3d}  loss {loss.item():.3f}  diag-acc {acc:.2f}")

    # ── Capture: deterministic center-window forward pass per clip ──────────────
    model.eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def dump_map(t: torch.Tensor) -> dict:
        """[B,C,H,W] (B=1) -> {shape:[H,W,C], data:[...]} channel-last flat."""
        a = t.detach()[0]
        if a.dim() == 1:                                # latent vector
            return {"shape": [a.shape[0]], "data": np.round(a.numpy(), 4).tolist()}
        a = a.permute(1, 2, 0).contiguous()             # H,W,C
        return {"shape": list(a.shape), "data": np.round(a.numpy().ravel(), 4).tolist()}

    clips_out = []
    windows_out = []
    audio_embeds, text_embeds = [], []
    with torch.no_grad():
        for slug, f, cap, p in zip(slugs, files, captions, paths):
            wav = load_window(p, center=True)
            windows_out.append({"slug": slug, "samples": np.round(wav, 5).tolist()})
            img = wav_to_mel_image(wav)[None]           # [1,1,IMG,IMG]
            cap_a: dict = {}
            za = model.embed_audio(img, cap_a)
            audio_embeds.append(za[0])

            # waveform preview, downsampled for display
            step = max(1, len(wav) // 800)
            wav_preview = np.round(wav[::step], 4).tolist()

            clips_out.append({
                "slug": slug,
                "file": f,
                "caption": cap,
                "waveform": wav_preview,
                "mel": dump_map(img),                   # the "image"
                "stages": {
                    "patch":  dump_map(cap_a["patch"]),    # 16x16x8
                    "stage1": dump_map(cap_a["stage1"]),   # 8x8x16
                    "stage2": dump_map(cap_a["stage2"]),   # 4x4x32
                    "stage3": dump_map(cap_a["stage3"]),   # 2x2x64
                },
                "latent": dump_map(cap_a["latent"]),    # 64
                "audioEmbed": np.round(za[0].numpy(), 4).tolist(),  # 512
            })

        text_out = []
        for slug, cap in zip(slugs, captions):
            ids = torch.tensor([encode_text(cap, vocab, max_len)])
            cap_t: dict = {}
            zt = model.embed_text(ids, cap_t)
            text_embeds.append(zt[0])
            toks = [t for t in re.findall(r"[a-z]+", cap.lower())]
            tok_ids = [vocab[t] for t in toks]
            text_out.append({
                "slug": slug,
                "caption": cap,
                "tokens": toks,
                "ids": tok_ids,
                "rawEmbed": {                                # embedding-table lookup [T,64]
                    "shape": [len(toks), LATENT],
                    "data": np.round(cap_t["raw"][0, :len(toks)].numpy().ravel(), 4).tolist(),
                },
                "tokenEmbed": {                              # after context mixing [T,64]
                    "shape": [len(toks), LATENT],
                    "data": np.round(cap_t["tokens"][0, :len(toks)].numpy().ravel(), 4).tolist(),
                },
                "latent": np.round(cap_t["latent"][0].numpy(), 4).tolist(),
                "textEmbed": np.round(zt[0].numpy(), 4).tolist(),
            })

        # NOTE: the opening-sphere data (hero.json) is produced by toy/hero_field.py
        # from REAL LAION-CLAP embeddings, on a different schema ({points, labels}).
        # This script must NOT write hero.json or it would clobber that file with an
        # incompatible shape the front-end can't read.

        za_all = torch.stack(audio_embeds)
        zt_all = torch.stack(text_embeds)
        sim = (za_all @ zt_all.t()).numpy()
        scale = model.logit_scale.exp().clamp(max=100).item()
        probs = torch.softmax(torch.tensor(sim) * scale, dim=1).numpy()

    meta = {
        "schema": 1,
        "model": "toy-clap",
        "embed_dim": EMBED, "latent_dim": LATENT, "mel_bins": MEL_BINS, "img": IMG,
        "sample_rate": SR, "win_sec": WIN_SEC,
        "stages": [
            {"name": "mel",    "shape": [IMG, IMG, 1],  "note": "log-mel image (real CLAP: 64xT -> 256x256)"},
            {"name": "patch",  "shape": [16, 16, 8],    "note": "patch embed (HTSAT: 64x64x96)"},
            {"name": "stage1", "shape": [8, 8, 16],     "note": "Swin stage 1 (HTSAT: 32x32x192)"},
            {"name": "stage2", "shape": [4, 4, 32],     "note": "Swin stage 2 (HTSAT: 16x16x384)"},
            {"name": "stage3", "shape": [2, 2, 64],     "note": "Swin stage 3/4 (HTSAT: 8x8x768)"},
            {"name": "latent", "shape": [64],           "note": "pooled (HTSAT: 768)"},
            {"name": "embed",  "shape": [EMBED],        "note": "projected + L2-normalized (CLAP: 512)"},
        ],
        "vocab": vocab,
        "logit_scale": round(scale, 4),
    }

    # ── Operands for the per-step computation explainer ─────────────────────
    # Real kernels / weights / filterbank so the "how stage k became stage k+1"
    # animation uses numbers that reconcile with the activations above.
    def t2d(t: torch.Tensor) -> dict:
        a = t.detach().cpu().numpy().astype(np.float64)  # round in f64 → short JSON reprs
        return {"shape": list(a.shape), "data": np.round(a.ravel(), 4).tolist()}

    convs = {
        "patch": model.audio.patch,
        "stage1": model.audio.s1,
        "stage2": model.audio.s2,
        "stage3": model.audio.s3,
    }
    mel_fb = mel_tf.mel_scale.fb  # [n_freqs, n_mels]
    weights = {
        "conv": {k: t2d(m.weight) for k, m in convs.items()},      # [out,in,3,3]
        "convBias": {k: t2d(m.bias) for k, m in convs.items()},
        "proj1": t2d(model.audio_proj.l1.weight),                  # [512,64]
        "proj1Bias": t2d(model.audio_proj.l1.bias),
        "proj2": t2d(model.audio_proj.l2.weight),                  # [512,512]
        "proj2Bias": t2d(model.audio_proj.l2.bias),
        "melfb": t2d(mel_fb),                                      # [n_freqs, n_mels]
        "embTable": t2d(model.text.emb.weight),                    # [vocab, 64] word → row
        # everything the in-browser forward pass needs to reproduce this run
        "mel": {"n_fft": 512, "hop": 256, "win": 512, "window": "hann",
                "center": True, "power": 2.0, "n_mels": MEL_BINS,
                "f_min": 50, "f_max": 7500, "sample_rate": SR,
                "log_eps": 1e-6, "standardize": True, "resize": IMG},
        "stride": 2,
        "kernel": 3,
    }
    (OUT_DIR / "weights.json").write_text(json.dumps(weights))

    (OUT_DIR / "meta.json").write_text(json.dumps(meta))
    # The browser recomputes every audio stage live from the weights, so clips.json
    # is just the demo index (slug/file/caption); full activations stay server-side.
    clips_slim = [{"slug": c["slug"], "file": c["file"], "caption": c["caption"]} for c in clips_out]
    (OUT_DIR / "clips.json").write_text(json.dumps(clips_slim))
    (OUT_DIR / "text.json").write_text(json.dumps(text_out))
    (OUT_DIR / "similarity.json").write_text(json.dumps({
        "slugs": slugs, "captions": captions,
        "cosine": np.round(sim, 4).tolist(),
        "probs": np.round(probs, 4).tolist(),
    }))
    # NOTE: clips_full.json and train_trajectory.json are intentionally NOT written —
    # the browser recomputes every stage live, so they were dead weight in the payload.

    total = sum(len(json.dumps(x)) for x in [clips_out, text_out, meta, traj])
    print(f"\nwrote {OUT_DIR}")
    print(f"  diagonal cosine: {np.round(np.diag(sim), 3).tolist()}")
    print(f"  off-diag mean:   {np.round((sim.sum()-np.trace(sim))/(N*N-N), 3)}")
    print(f"  approx JSON bytes: {total:,}")


if __name__ == "__main__":
    main()
