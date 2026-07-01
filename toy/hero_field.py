"""
hero_field.py — the opening sphere's data: REAL LAION-CLAP embeddings.

The step-by-step post uses a miniature encoder, but the hero should show the real
thing at scale: 16,441 actual CLAP embeddings of real car-fault sounds (from the
detect-mech-issues corpus), each a unit 512-d vector — so they genuinely live on a
sphere. We sample a few thousand, project to 3-D (PCA, axis-balanced), and ship
their sphere positions + a colour scalar. No labels, no explanation: it's the field
of "every car sound CLAP has ever heard," and it should just look alive.

Run:  .venv/bin/python toy/hero_field.py
Out:  post/public/data/hero.json  ([{xyz:[3], c:0..1}])
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = Path.home() / "Projects/detect-mech-issues/training1/data/clap_embeddings.npz"
OUT = ROOT / "post/public/data/hero.json"
N = 2600


def kmeans(X: np.ndarray, K: int, iters: int = 40, seed: int = 0) -> np.ndarray:
    """Plain Lloyd's k-means (no sklearn dependency)."""
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), K, replace=False)].copy()
    a = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)     # [N,K]
        a = d.argmin(1)
        for k in range(K):
            m = X[a == k]
            if len(m):
                C[k] = m.mean(0)
    return a

def main():
    d = np.load(SRC, allow_pickle=True)
    X = d["X"].astype(np.float64)                       # [16441, 512], unit-norm
    ids = [str(s) for s in d["ids"]]
    rng = np.random.RandomState(7)
    idx = rng.choice(len(X), min(N, len(X)), replace=False)
    Xs = X[idx]
    ids_s = [ids[i] for i in idx]                       # keep the source of every point

    # one PCA transform, reused for the cloud AND the labelled centroids
    mean = Xs.mean(0)
    _, _, Wt = np.linalg.svd(Xs - mean, full_matrices=False)
    axstd = ((Xs - mean) @ Wt[:3].T).std(0)

    def to_sphere(V: np.ndarray) -> np.ndarray:
        P = ((V - mean) @ Wt[:3].T) / (axstd + 1e-9)    # balance the 3 axes
        return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)

    P = to_sphere(Xs)

    # colour by GROUP: k-means over a richer subspace so similar sounds share a hue
    F = (Xs - mean) @ Wt[:16].T
    g = kmeans(F, 7, seed=0)
    c4 = (Xs - mean) @ Wt[3]
    c = (c4 - c4.min()) / (c4.max() - c4.min() + 1e-9)

    pts = [{"xyz": [round(float(v), 3) for v in P[i]], "c": round(float(c[i]), 3),
            "g": int(g[i]), "id": ids_s[i]}
           for i in range(len(P))]

    # labelled landmarks: the centroid of each named fault, placed in the same space
    FAULTS = [("low_oil", "low oil"), ("power_steering", "power steering"),
              ("serpentine_belt", "serpentine belt"), ("worn_out_brakes", "worn brakes"),
              ("bad_ignition", "bad ignition"), ("dead_battery", "dead battery"),
              ("normal_engine_idle", "normal idle")]
    labels = []
    for fault, name in FAULTS:
        rows = [i for i, s in enumerate(ids) if s.startswith(f"cardiag:{fault}:")]
        if not rows:
            continue
        xyz = to_sphere(X[rows].mean(0, keepdims=True))[0]
        labels.append({"label": name, "xyz": [round(float(v), 3) for v in xyz], "n": len(rows)})

    OUT.write_text(json.dumps({"points": pts, "labels": labels}))
    print(f"wrote {len(pts)} points + {len(labels)} labels → {OUT}")
    print(f"  per-axis std: {[round(float(s), 3) for s in P.std(0)]}")

if __name__ == "__main__":
    main()
