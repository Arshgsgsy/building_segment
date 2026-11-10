"""Sheet loading and paper normalization (guide §5.3). Later: tile sampler, augmentation, stain bank (§5.2, §5.4)."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sanborn import ROOT

Image.MAX_IMAGE_PIXELS = None  # full LOC sheets are ~50 MP
CACHE = ROOT / "data" / "cache"


def load_sheet(path):
    """RGB uint8 array. A JP2 is decoded once and cached as .npy; later loads are memory-mapped and instant."""
    path = Path(path)
    if path.suffix.lower() != ".jp2":
        return np.asarray(Image.open(path).convert("RGB"))
    cache = CACHE / f"{path.stem}.npy"
    if not cache.exists():
        print(f"decoding {path.name} once, caching to {cache.relative_to(ROOT)}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, np.asarray(Image.open(path).convert("RGB")))
    return np.load(cache, mmap_mode="r")


def paper_color(img, step=8):
    """Median RGB of bright, low-chroma pixels, estimated on every `step`-th pixel."""
    sub = np.ascontiguousarray(img[::step, ::step]).reshape(1, -1, 3)
    lab = cv2.cvtColor(sub.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[0]
    L, C = lab[:, 0], np.hypot(lab[:, 1], lab[:, 2])
    bright = L > np.percentile(L, 50)
    pale = C <= np.percentile(C[bright], 50)
    return np.median(sub[0][bright & pale], axis=0)


def normalize(img, paper):
    """Divide by the paper color so paper becomes white. Returns float32 RGB in [0, 1]."""
    out = img.astype(np.float32) / np.maximum(paper, 1).astype(np.float32)
    return np.clip(out, 0, 1, out=out)
