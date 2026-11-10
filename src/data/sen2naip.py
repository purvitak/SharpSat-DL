"""
SEN2NAIP cross-sensor pairs: real Sentinel-2 L2A (10 m) LR images matched to
NAIP aerial HR images (2.5 m) -- an exact x4 with the same R, G, B, NIR bands.
https://huggingface.co/datasets/isp-uv-es/SEN2NAIP (CC-BY-4.0, Aybar et al. 2024)

Unlike the rest of src/data, the LR side is NOT made by degrade(): it is a real
satellite image, so a model trained here learns actual Sentinel-2 -> 2.5 m SR.

download() reads only the requested pairs out of the 2.2 GB remote zip using
HTTP range requests, so taking 1,500 of the 2,851 pairs fetches ~1 GB.

Radiometric harmonisation. LR is surface reflectance (DN / 10000); NAIP is an
8-bit camera product (DN / 255) on a different, scene-dependent scale. Left as
is, LR and HR disagree on brightness, which breaks the model's bicubic skip and
the NDVI loss. Each NAIP band is therefore mapped linearly so that its 4x4
block average matches the S2 band's mean and std:

    hr = gain * (naip / 255) + offset

The gains/offsets are stored in index.json, not baked into the files, so the
raw data stays untouched.

Quality filter. NAIP and S2 are acquired on different days, so some pairs show
clouds in S2 or real change between dates. A pair is dropped if the Pearson
correlation between the block-averaged NAIP and S2 (band mean) is below
`min_corr`, or if either image has nodata.

Layout:
    data/sen2naip/ROI_XXXX_lr.npy   (121, 121, 4) uint16  S2 reflectance x 10000
    data/sen2naip/ROI_XXXX_hr.npy   (484, 484, 4) uint8   NAIP DN
    data/sen2naip/index.json        split, gain, offset, corr per pair

Usage:
    python -m src.data.sen2naip --pairs 1500
"""

import argparse
import io
import json
import random
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ZIP_URL = ("https://huggingface.co/datasets/isp-uv-es/SEN2NAIP/resolve/main/"
           "cross-sensor/cross-sensor.zip")
SCALE = 4
LR_SIZE, HR_SIZE = 121, 484


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _resolve(url):
    """HF redirects to a signed CDN URL; resolve once so each range read is one request."""
    try:
        resp = urllib.request.build_opener(_NoRedirect).open(urllib.request.Request(url, method="HEAD"))
        return url, int(resp.headers["Content-Length"])
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise
        final = e.headers["Location"]
    size = int(urllib.request.urlopen(urllib.request.Request(final, method="HEAD"))
               .headers["Content-Length"])
    return final, size


class _HTTPRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, so zipfile can read single members."""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buf):
        end = min(self.size, self.pos + len(buf))
        if end <= self.pos:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        for attempt in range(4):
            try:
                data = urllib.request.urlopen(req, timeout=60).read()
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 3:
                    raise
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _open_zip(url, size):
    return zipfile.ZipFile(io.BufferedReader(_HTTPRangeFile(url, size), buffer_size=1 << 20))


def _read_tif(blob):
    from rasterio.io import MemoryFile
    with MemoryFile(blob) as mem, mem.open() as src:
        return src.read().transpose(1, 2, 0)   # (H, W, C)


def harmonise(lr_dn, hr_dn):
    """Per-band linear map of NAIP onto S2 reflectance; returns gain, offset, corr."""
    lr = lr_dn.astype(np.float64) / 10000.0
    hr = hr_dn.astype(np.float64) / 255.0
    hr_ds = hr.reshape(LR_SIZE, SCALE, LR_SIZE, SCALE, -1).mean(axis=(1, 3))
    gain = lr.std(axis=(0, 1)) / np.maximum(hr_ds.std(axis=(0, 1)), 1e-6)
    offset = lr.mean(axis=(0, 1)) - gain * hr_ds.mean(axis=(0, 1))
    corr = float(np.corrcoef(hr_ds.mean(-1).ravel(), lr.mean(-1).ravel())[0, 1])
    return gain.tolist(), offset.tolist(), corr


def _fetch_chunk(url, size, rois, out_dir, min_corr):
    zf = _open_zip(url, size)
    kept = []
    for roi in rois:
        lr = _read_tif(zf.read(f"cross-sensor/{roi}/lr.tif"))
        hr = _read_tif(zf.read(f"cross-sensor/{roi}/hr.tif"))
        if lr.shape != (LR_SIZE, LR_SIZE, 4) or hr.shape != (HR_SIZE, HR_SIZE, 4):
            continue
        if (lr <= 0).mean() > 0.001 or (hr.max(axis=-1) == 0).mean() > 0.001:
            continue
        gain, offset, corr = harmonise(lr, hr)
        if not np.isfinite(corr) or corr < min_corr:
            continue
        np.save(out_dir / f"{roi}_lr.npy", np.clip(lr, 0, 65535).astype(np.uint16))
        np.save(out_dir / f"{roi}_hr.npy", hr.astype(np.uint8))
        kept.append({"roi": roi, "gain": gain, "offset": offset, "corr": round(corr, 4)})
    return kept


def download(pairs=1500, out_dir="data/sen2naip", min_corr=0.6, val_frac=0.1,
             test_frac=0.1, seed=0, workers=8):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    url, size = _resolve(ZIP_URL)
    rois = sorted({n.split("/")[1] for n in _open_zip(url, size).namelist()
                   if n.endswith("/hr.tif")})
    print(f"SEN2NAIP cross-sensor: {len(rois)} pairs available, taking {pairs}")
    random.Random(seed).shuffle(rois)

    # Draw in rounds until enough pairs pass the quality filter.
    kept, cursor = [], 0
    with ThreadPoolExecutor(workers) as pool:
        while len(kept) < pairs and cursor < len(rois):
            want = int((pairs - len(kept)) * 1.15) + workers
            batch, cursor = rois[cursor:cursor + want], cursor + want
            chunks = [batch[i::workers] for i in range(workers)]
            for res in pool.map(lambda c: _fetch_chunk(url, size, c, out_dir, min_corr), chunks):
                kept.extend(res)
            print(f"  fetched {cursor} of {len(rois)}, {len(kept)} passed the quality filter")
    kept = sorted(kept, key=lambda p: p["roi"])[:pairs]
    for extra in set(rois[:cursor]) - {p["roi"] for p in kept}:
        for side in ("lr", "hr"):
            (out_dir / f"{extra}_{side}.npy").unlink(missing_ok=True)

    # Split by ROI: each ROI is a separate 1.2 km site, so no crop leaks across splits.
    order = list(range(len(kept)))
    random.Random(seed + 1).shuffle(order)
    n_test, n_val = round(len(kept) * test_frac), round(len(kept) * val_frac)
    for rank, i in enumerate(order):
        kept[i]["split"] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"

    with open(out_dir / "index.json", "w") as f:
        json.dump({"source": ZIP_URL, "min_corr": min_corr, "pairs": kept}, f, indent=0)
    counts = {s: sum(p["split"] == s for p in kept) for s in ("train", "val", "test")}
    print(f"{len(kept)} pairs -> {out_dir}/  ({counts})")
    return counts


class SEN2NAIPDataset(Dataset):
    """
    (LR, HR) crops: LR (4, lr_crop, lr_crop) real S2, HR (4, 4*lr_crop, 4*lr_crop)
    harmonised NAIP, both in reflectance. Same output shapes as SatelliteSRDataset.

    train: random aligned crop of a random pair on every __getitem__ (`length` per
    epoch). val / test: the fixed 3x3 grid of non-overlapping crops per pair.
    """

    def __init__(self, split, root="data/sen2naip", augment=None, length=2400, lr_crop=32):
        root = Path(root)
        if not (root / "index.json").exists():
            raise FileNotFoundError(f"{root}/index.json not found -- run "
                                    f"`python -m src.data.sen2naip` first.")
        with open(root / "index.json") as f:
            self.pairs = [p for p in json.load(f)["pairs"] if p["split"] == split]
        if not self.pairs:
            raise ValueError(f"No '{split}' pairs in {root}/index.json")
        self.root = root
        self.random_crops = split == "train"
        self.augment = self.random_crops if augment is None else augment
        self.length = length
        self.lr_crop = lr_crop
        self.crops = [(i, y, x) for i in range(len(self.pairs))
                      for y in range(0, LR_SIZE - lr_crop + 1, lr_crop)
                      for x in range(0, LR_SIZE - lr_crop + 1, lr_crop)]

    def __len__(self):
        return self.length if self.random_crops else len(self.crops)

    def _load(self, i, y, x):
        p = self.pairs[i]
        c, hc = self.lr_crop, self.lr_crop * SCALE
        lr = np.load(self.root / f"{p['roi']}_lr.npy", mmap_mode="r")[y:y + c, x:x + c]
        hr = np.load(self.root / f"{p['roi']}_hr.npy", mmap_mode="r")[
            y * SCALE:y * SCALE + hc, x * SCALE:x * SCALE + hc]
        lr = lr.astype(np.float32) / 10000.0
        hr = hr.astype(np.float32) / 255.0 * np.float32(p["gain"]) + np.float32(p["offset"])
        return np.clip(lr, 0, 1), np.clip(hr, 0, 1)

    def __getitem__(self, idx):
        if self.random_crops:
            i = torch.randint(len(self.pairs), (1,)).item()
            y, x = torch.randint(0, LR_SIZE - self.lr_crop + 1, (2,)).tolist()
            lr, hr = self._load(i, y, x)
        else:
            lr, hr = self._load(*self.crops[idx])

        if self.augment:
            # identical rotation/flip on both sides keeps them aligned
            k, flip_h, flip_v = torch.randint(0, 4, (3,)).tolist()
            lr, hr = np.rot90(lr, k, axes=(0, 1)), np.rot90(hr, k, axes=(0, 1))
            if flip_h % 2:
                lr, hr = lr[:, ::-1], hr[:, ::-1]
            if flip_v % 2:
                lr, hr = lr[::-1, :], hr[::-1, :]

        return (torch.from_numpy(np.ascontiguousarray(lr.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(hr.transpose(2, 0, 1))))


def export_web_demo(n=8, root="data/sen2naip", out_dir="web/data/pairs", min_corr=0.8):
    """
    Writes `n` held-out TEST pairs for the web dashboard as <roi>.npz with
    lr (121, 121, 4) S2 reflectance and hr (484, 484, 4) harmonised NAIP. The
    most textured well-registered sites are chosen, since flat fields show
    nothing; only test sites are used so the demo never shows training data.
    """
    root, out_dir = Path(root), Path(out_dir)
    with open(root / "index.json") as f:
        pairs = [p for p in json.load(f)["pairs"] if p["split"] == "test" and p["corr"] >= min_corr]
    texture = []
    for p in pairs:
        lr = np.load(root / f"{p['roi']}_lr.npy").astype(np.float32) / 10000.0
        hr = np.load(root / f"{p['roi']}_hr.npy").astype(np.float32) / 255.0
        hr = np.clip(hr * np.float32(p["gain"]) + np.float32(p["offset"]), 0, 1)
        # mean absolute vertical gradient as a cheap texture score
        texture.append((float(np.abs(np.diff(hr.mean(-1), axis=0)).mean()), p, np.clip(lr, 0, 1), hr))
    texture.sort(key=lambda t: t[0], reverse=True)
    # every other one of the most textured sites, so the set isn't all one land-cover type
    chosen = texture[: 2 * n : 2]
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, p, lr, hr in chosen:
        np.savez_compressed(out_dir / f"{p['roi']}.npz", lr=lr, hr=hr)
    print(f"{len(chosen)} held-out test pairs -> {out_dir}/: {', '.join(c[1]['roi'] for c in chosen)}")
    return [c[1]["roi"] for c in chosen]


def get_sen2naip_dataloaders(batch_size=8, root="data/sen2naip", train_length=2400,
                             num_workers=2):
    train_ds = SEN2NAIPDataset("train", root=root, length=train_length)
    val_ds = SEN2NAIPDataset("val", root=root, augment=False)
    # train crops are already random, so shuffling the indices adds nothing
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=1500,
                        help="how many of the 2,851 pairs to keep (after quality filtering)")
    parser.add_argument("--out_dir", default="data/sen2naip")
    parser.add_argument("--min_corr", type=float, default=0.6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--export_web", type=int, default=0, metavar="N",
                        help="instead of downloading, write N held-out test pairs to web/data/pairs/")
    args = parser.parse_args()
    if args.export_web:
        export_web_demo(args.export_web, root=args.out_dir)
    else:
        download(args.pairs, args.out_dir, args.min_corr, workers=args.workers)
