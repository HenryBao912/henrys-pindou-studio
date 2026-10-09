#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Henry's Pindou Studio - 给田田田的拼豆图纸
Turn a photo into a fuse-bead (拼豆) pattern, 100% offline.

PRIVACY
  This program contains no networking code. Photos are read from your disk,
  processed in memory, and written only to the files you choose when exporting.

SETUP (once)
  pip install pillow numpy reportlab
  optional: pip install pillow-heif     (open iPhone .HEIC photos directly)

RUN
  python henrys_pindou_studio.py            (or: python henrys_pindou_studio.py photo.jpg)
  python henrys_pindou_studio.py photo.jpg --export out/cat --width 52   (no window)

Bead palettes are bundled from the open datasets HansBug/pindou-color-data and
maxcleme/beadcolors, as packaged by the Apache-2.0 project HansBug/pypindou.
Screen colors approximate real beads, so check against your physical color card.
"""

import csv
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import date

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
except ImportError as _exc:  # pragma: no cover
    sys.stderr.write("Missing library: %s\nInstall with:  pip install pillow numpy reportlab\n" % _exc)
    raise SystemExit(1)

try:  # optional: iPhone HEIC photos
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_OK = True
except Exception:
    HEIF_OK = False

APP_NAME = "Henry's Pindou Studio"
APP_SUBTITLE = "给田田田的拼豆图纸"
APP_VERSION = "1.1"


def _resample(name):
    holder = getattr(Image, "Resampling", Image)
    return getattr(holder, name)


BOX = _resample("BOX")
LANCZOS = _resample("LANCZOS")
BICUBIC = _resample("BICUBIC")

MAX_WORK_SIDE = 1600          # photos are downscaled to this for processing (plenty for 200+ beads)
EMPTY_RGB = (238, 238, 236)   # "no bead" cells in flat previews
BOARD_RGB = (226, 228, 230)   # pegboard color in bead previews
PEG_RGB = (200, 203, 207)
HOLE_RGB = (196, 198, 202)

BEAD_TYPES = [
    ("Mini 2.6 mm (MARD, COCO, Artkal C...)", 2.6, 52),
    ("Hama Mini 2.5 mm", 2.5, None),
    ("Midi 5 mm (Perler, Hama, Artkal S)", 5.0, 29),
    ("Maxi 10 mm", 10.0, None),
]
CHART_MODES = ["One page per board", "Whole pattern on one page", "Large print (split boards)"]
CLEAN_NAMES = {0: "Off", 1: "Light: merge specks", 2: "Medium", 3: "Strong: big clean areas"}
ASPECTS = {"Free": None, "1:1": 1.0, "3:4 portrait": 3 / 4, "4:3 landscape": 4 / 3, "2:3 portrait": 2 / 3}
PAPERS = {"Letter": (612.0, 792.0), "A4": (595.2756, 841.8898)}
MM = 72.0 / 25.4  # PDF points per millimetre


# =============================================================================
# Color science
# =============================================================================
_M_SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]])
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb):
    """sRGB (0-255, any shape ending in 3) to CIE Lab (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _M_SRGB_TO_XYZ.T) / _WHITE_D65
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _de2000_block(lab1, lab2):
    L1, a1, b1 = lab1[:, 0:1], lab1[:, 1:2], lab1[:, 2:3]
    L2, a2, b2 = lab2[None, :, 0], lab2[None, :, 1], lab2[None, :, 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar7 = ((C1 + C2) / 2.0) ** 7
    G = 0.5 * (1.0 - np.sqrt(Cbar7 / (Cbar7 + 25.0 ** 7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    prod = C1p * C2p
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where(prod == 0, 0.0, dhp)
    dHp = 2.0 * np.sqrt(prod) * np.sin(np.radians(dhp) / 2.0)
    Lbarp = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0
    hsum = h1p + h2p
    habs = np.abs(h1p - h2p)
    hbarp = np.where(prod == 0, hsum,
                     np.where(habs <= 180.0, hsum / 2.0,
                              np.where(hsum < 360.0, (hsum + 360.0) / 2.0, (hsum - 360.0) / 2.0)))
    T = (1.0 - 0.17 * np.cos(np.radians(hbarp - 30.0)) + 0.24 * np.cos(np.radians(2.0 * hbarp))
         + 0.32 * np.cos(np.radians(3.0 * hbarp + 6.0)) - 0.20 * np.cos(np.radians(4.0 * hbarp - 63.0)))
    dtheta = 30.0 * np.exp(-(((hbarp - 275.0) / 25.0) ** 2))
    Cbarp7 = Cbarp ** 7
    Rc = 2.0 * np.sqrt(Cbarp7 / (Cbarp7 + 25.0 ** 7))
    Sl = 1.0 + (0.015 * (Lbarp - 50.0) ** 2) / np.sqrt(20.0 + (Lbarp - 50.0) ** 2)
    Sc = 1.0 + 0.045 * Cbarp
    Sh = 1.0 + 0.015 * Cbarp * T
    Rt = -np.sin(np.radians(2.0 * dtheta)) * Rc
    tl, tc, th = dLp / Sl, dCp / Sc, dHp / Sh
    return np.sqrt(np.maximum(tl * tl + tc * tc + th * th + Rt * tc * th, 0.0))


def delta_e(lab1, lab2):
    """CIEDE2000 distance matrix between two lists of Lab colors: shape (len1, len2)."""
    lab1 = np.atleast_2d(np.asarray(lab1, dtype=np.float64))
    lab2 = np.atleast_2d(np.asarray(lab2, dtype=np.float64))
    out = np.empty((len(lab1), len(lab2)))
    chunk = max(256, 1_500_000 // max(1, len(lab2)))
    for i in range(0, len(lab1), chunk):
        out[i:i + chunk] = _de2000_block(lab1[i:i + chunk], lab2)
    return out


def luminance(rgb):
    r, g, b = (float(v) / 255.0 for v in rgb[:3])
    return 0.299 * r + 0.587 * g + 0.114 * b


def ink_for(rgb):
    """Black or white text, whichever reads better on this color."""
    return (0, 0, 0) if luminance(rgb) > 0.55 else (255, 255, 255)


# =============================================================================
# Palettes
# =============================================================================
class Palette:
    def __init__(self, key, data):
        self.key = key
        self.label = data["label"]
        self.name = data["name"]
        self.mm = float(data["mm"])
        rows = data["colors"]
        self.codes = [r[0] for r in rows]
        self.names = [r[1] or r[0] for r in rows]
        self.hex = [r[2] for r in rows]
        self.special = np.array([bool(r[3]) for r in rows])
        self.rgb = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in self.hex], dtype=np.uint8)
        self.lab = srgb_to_lab(self.rgb)
        self.index = {c.upper(): i for i, c in enumerate(self.codes)}

    def __len__(self):
        return len(self.codes)

    def describe(self, idx):
        code, name = self.codes[idx], self.names[idx]
        return code if name == code else "%s %s" % (code, name)


_PALETTES = None


def palettes():
    global _PALETTES
    if _PALETTES is None:
        data = json.loads(PALETTE_JSON)
        _PALETTES = {k: Palette(k, v) for k, v in data.items()}
    return _PALETTES


def default_board_for(pitch_mm):
    return 52 if pitch_mm < 4 else 29


# =============================================================================
# Settings and result
# =============================================================================
@dataclass
class Settings:
    palette: str = "mard221"
    width: int = 52
    pitch_mm: float = 2.6
    board: int = 52
    max_colors: int = 16
    method: str = "dominant"            # "dominant" (clean edges) or "average" (smooth shading)
    detail: int = 2                     # 0-3: keep thin dark features (eyes, brows, lines) in dominant mode
    rotation: float = 0.0               # degrees, counter-clockwise
    flip: bool = False
    crop: tuple = (0.0, 0.0, 1.0, 1.0)  # normalized to the rotated image
    offset_x: float = 0.0               # grid nudge, in beads
    offset_y: float = 0.0
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    remove_bg: bool = False
    bg_tolerance: float = 12.0
    cleanup: int = 1
    outline: bool = False
    outline_code: str = ""
    only_codes: str = ""
    exclude_codes: str = ""
    allow_special: bool = False


@dataclass
class Pattern:
    grid: np.ndarray        # (H, W) palette indices, -1 = no bead
    pal: Palette
    pitch_mm: float = 2.6
    board: int = 52
    notes: list = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def H(self):
        return int(self.grid.shape[0])

    @property
    def W(self):
        return int(self.grid.shape[1])

    def total(self):
        return int((self.grid >= 0).sum())

    def used_sorted(self, grid=None):
        g = self.grid if grid is None else grid
        g = g[g >= 0]
        if g.size == 0:
            return []
        idx, cnt = np.unique(g, return_counts=True)
        order = np.lexsort((idx, -cnt))
        return [(int(idx[i]), int(cnt[i])) for i in order]

    def boards(self):
        return math.ceil(self.W / self.board), math.ceil(self.H / self.board)

    def size_text(self):
        w_cm, h_cm = self.W * self.pitch_mm / 10.0, self.H * self.pitch_mm / 10.0
        return "%.1f x %.1f cm (%.1f x %.1f in)" % (w_cm, h_cm, w_cm / 2.54, h_cm / 2.54)


def pattern_labels(pat):
    """Short text printed inside each cell: the color code, or a number if codes are long."""
    used = [i for i, _ in pat.used_sorted()]
    if not used:
        return {}
    if max(len(pat.pal.codes[i]) for i in used) <= 4:
        return {i: pat.pal.codes[i] for i in used}
    return {i: str(n + 1) for n, i in enumerate(used)}


# =============================================================================
# Image preparation
# =============================================================================
def load_photo(path):
    img = Image.open(path)
    try:
        img = ImageOps.exif_transpose(img)  # phone photos store rotation in EXIF
    except Exception:
        pass
    img = img.convert("RGBA")
    w, h = img.size
    s = MAX_WORK_SIDE / float(max(w, h))
    if s < 1.0:
        size = (max(1, round(w * s)), max(1, round(h * s)))
        img = img.convert("RGBa").resize(size, LANCZOS).convert("RGBA")
    return img


def transform_source(img, rotation=0.0, flip=False):
    if flip:
        img = ImageOps.mirror(img)
    if abs(rotation) > 0.01:
        img = (img.convert("RGBa")
               .rotate(rotation, resample=BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))
               .convert("RGBA"))
    return img


def adjust_colors(img, brightness=1.0, contrast=1.0, saturation=1.0):
    if max(abs(brightness - 1), abs(contrast - 1), abs(saturation - 1)) < 1e-3:
        return img
    alpha = img.getchannel("A")
    rgb = img.convert("RGB")
    if abs(brightness - 1) >= 1e-3:
        rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    if abs(contrast - 1) >= 1e-3:
        rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    if abs(saturation - 1) >= 1e-3:
        rgb = ImageEnhance.Color(rgb).enhance(saturation)
    rgb.putalpha(alpha)
    return rgb


def rotated_size(w, h, deg):
    t = math.radians(deg)
    c, s = abs(math.cos(t)), abs(math.sin(t))
    return w * c + h * s, w * s + h * c


def remap_point_rotation(px, py, w, h, deg_from, deg_to):
    """Point in the canvas of an image rotated by deg_from -> same content point at deg_to."""
    W1, H1 = rotated_size(w, h, deg_from)
    W2, H2 = rotated_size(w, h, deg_to)
    dx, dy = px - W1 / 2.0, py - H1 / 2.0
    t = math.radians(deg_from)
    c, s = math.cos(t), math.sin(t)
    sx, sy = c * dx - s * dy, s * dx + c * dy           # back to the unrotated image
    t2 = math.radians(deg_to)
    c2, s2 = math.cos(t2), math.sin(t2)
    nx, ny = c2 * sx + s2 * sy, -s2 * sx + c2 * sy       # forward (counter-clockwise, y down)
    return nx + W2 / 2.0, ny + H2 / 2.0


def grid_geometry(img_w, img_h, crop, width, offset_x=0.0, offset_y=0.0):
    """Square bead cells laid over the crop. Never stretches: height follows the crop's aspect."""
    x0, y0, x1, y1 = crop
    x0, x1 = x0 * img_w, x1 * img_w
    y0, y1 = y0 * img_h, y1 * img_h
    cw, ch = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
    W = max(1, int(width))
    cs = cw / W
    H = max(1, int(round(ch / cs)))
    cy = (y0 + y1) / 2.0
    gx0 = x0 + offset_x * cs
    gy0 = cy - H * cs / 2.0 + offset_y * cs
    return W, H, cs, gx0, gy0


# =============================================================================
# Pattern engine
# =============================================================================
def parse_codes(text, pal):
    found, unknown = [], []
    for tok in re.split(r"[\s,;，；、]+", (text or "").strip()):
        if not tok:
            continue
        i = pal.index.get(tok.upper())
        if i is None:
            unknown.append(tok)
        else:
            found.append(i)
    return found, unknown


def allowed_indices(pal, s, notes):
    if s.only_codes.strip():
        found, unknown = parse_codes(s.only_codes, pal)
        mask = np.zeros(len(pal), bool)
        mask[found] = True
        if unknown:
            notes.append("Not in %s: %s" % (pal.name, ", ".join(unknown[:12])))
    else:
        mask = np.ones(len(pal), bool) if s.allow_special else ~pal.special
    if s.exclude_codes.strip():
        found, unknown = parse_codes(s.exclude_codes, pal)
        mask[found] = False
        if unknown:
            notes.append("Not in %s: %s" % (pal.name, ", ".join(unknown[:12])))
    if not mask.any():
        notes.append("No usable colors left after your filters, so using the full palette.")
        mask = ~pal.special
    return np.nonzero(mask)[0]


def unique_colors(rgb_flat):
    """Bucket colors to 5 bits per channel (invisible difference) to speed up matching."""
    q = (np.asarray(rgb_flat, dtype=np.int32) >> 3)
    key = (q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    rep = np.stack([((uniq >> 10) & 31) * 8 + 4, ((uniq >> 5) & 31) * 8 + 4, (uniq & 31) * 8 + 4], 1)
    return rep.astype(np.uint8), inv.reshape(-1), counts


def select_colors(sample_lab, weights, pal_lab, allowed, max_colors):
    """Pick at most max_colors real bead colors that represent the samples best.

    Start from every bead color that is the nearest match for something, then repeatedly
    drop the color whose removal adds the least squared error. Small but very different
    details (eyes, highlights) are expensive to drop, so they tend to survive."""
    if len(sample_lab) == 0 or len(allowed) == 0:
        return np.asarray(allowed[:1])
    D = delta_e(sample_lab, pal_lab[allowed])
    active = np.unique(D.argmin(1))
    D2 = D[:, active] ** 2
    max_colors = max(1, int(max_colors))
    weights = np.asarray(weights, dtype=np.float64)
    while len(active) > max_colors:
        A = len(active)
        best = D2.argmin(1)
        part = np.partition(D2, 1, axis=1)
        penalty = np.bincount(best, weights=weights * (part[:, 1] - part[:, 0]), minlength=A)
        excess = A - max_colors
        n_drop = 1 if excess <= 6 else max(1, excess // 3)
        drop = np.argsort(penalty, kind="stable")[:n_drop]
        keep = np.ones(A, bool)
        keep[drop] = False
        active = active[keep]
        D2 = D2[:, keep]
    return np.asarray(allowed)[active]


def flood_from_border(passable):
    H, W = passable.shape
    pas = passable.tolist()
    vis = [[False] * W for _ in range(H)]
    stack = []
    for y in range(H):
        for x in (0, W - 1):
            if pas[y][x] and not vis[y][x]:
                vis[y][x] = True
                stack.append((y, x))
    for x in range(W):
        for y in (0, H - 1):
            if pas[y][x] and not vis[y][x]:
                vis[y][x] = True
                stack.append((y, x))
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < H and 0 <= nx < W and pas[ny][nx] and not vis[ny][nx]:
                vis[ny][nx] = True
                stack.append((ny, nx))
    return np.array(vis, dtype=bool)


def detect_background(cell_lab, empty, tol):
    """Plain-background removal: flood fill from the edges through cells close to the
    dominant edge color."""
    H, W = empty.shape
    border = np.zeros((H, W), bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    sel = border & ~empty
    if not sel.any():
        return np.zeros((H, W), bool), None
    labs = cell_lab[sel]
    keys = np.round(labs / 7.0).astype(np.int64) + 64
    k1 = keys[:, 0] * 1_000_000 + keys[:, 1] * 1_000 + keys[:, 2]
    _, inv, cnt = np.unique(k1, return_inverse=True, return_counts=True)
    bg = labs[inv.reshape(-1) == cnt.argmax()].mean(0)
    d = delta_e(cell_lab.reshape(-1, 3), bg[None, :]).reshape(H, W)
    cand = d <= tol
    reached = flood_from_border(cand | empty)
    return reached & cand & ~empty, bg


def touches_empty(empty):
    """Cells with an empty (or off-pattern) 4-neighbour."""
    p = np.pad(empty, 1, constant_values=True)
    return p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]


def peel_halo(empty, bg_like, layers=2):
    """Remove the soft 'halo' ring where the subject's edge blends into the removed background."""
    for _ in range(layers):
        halo = bg_like & touches_empty(empty) & ~empty
        if not halo.any():
            break
        empty = empty | halo
    return empty


def label_components(filled):
    """4-connected pieces (beads that touch only at corners are separate pieces)."""
    H, W = filled.shape
    f = filled.tolist()
    lab = [[-1] * W for _ in range(H)]
    sizes = []
    for y in range(H):
        row = f[y]
        for x in range(W):
            if row[x] and lab[y][x] < 0:
                n = len(sizes)
                lab[y][x] = n
                stack = [(y, x)]
                size = 0
                while stack:
                    cy, cx = stack.pop()
                    size += 1
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < H and 0 <= nx < W and f[ny][nx] and lab[ny][nx] < 0:
                            lab[ny][nx] = n
                            stack.append((ny, nx))
                sizes.append(size)
    return np.array(lab, dtype=np.int64).reshape(H, W), sizes


def analyze_structure(grid):
    """Returns (piece sizes, corner-only joints). After ironing, pieces that are joined only
    diagonally (corner to corner) are weak and tend to snap apart."""
    filled = grid >= 0
    if not filled.any():
        return [], []
    labels, sizes = label_components(filled)
    a, b = labels[:-1, :-1], labels[1:, 1:]
    c, d = labels[:-1, 1:], labels[1:, :-1]
    diag1 = (a >= 0) & (b >= 0) & (c < 0) & (d < 0) & (a != b)
    diag2 = (c >= 0) & (d >= 0) & (a < 0) & (b < 0) & (c != d)
    pts = np.argwhere(diag1 | diag2)
    weak = [(int(r) + 1, int(cc) + 1) for r, cc in pts]   # corner coordinates (in cells)
    return sorted(sizes, reverse=True), weak


def remove_small_islands(grid, max_size):
    filled = grid >= 0
    labels, sizes = label_components(filled)
    if len(sizes) <= 1:
        return grid, 0
    biggest = int(np.argmax(sizes))
    small = [i for i, s in enumerate(sizes) if s <= max_size and i != biggest]
    if not small:
        return grid, 0
    g = grid.copy()
    g[np.isin(labels, small)] = -1
    return g, int(sum(sizes[i] for i in small))


CLEANUP_LEVELS = {1: (1, 15.0), 2: (2, 28.0), 3: (4, 1e9)}  # (max region size, max color distance)


def cleanup_grid(grid, pal, level, passes=2):
    """Merge tiny same-color specks into their surroundings (BFS over 4-connected regions)."""
    params = CLEANUP_LEVELS.get(int(level))
    if not params:
        return grid
    max_size, max_de = params
    used = np.unique(grid[grid >= 0])
    if len(used) < 2:
        return grid
    de = delta_e(pal.lab[used], pal.lab[used])
    pos = {int(c): i for i, c in enumerate(used)}
    H, W = grid.shape
    rows = grid.tolist()
    for _ in range(passes):
        changed = False
        seen = [[False] * W for _ in range(H)]
        for y in range(H):
            for x in range(W):
                col = rows[y][x]
                if col < 0 or seen[y][x]:
                    continue
                seen[y][x] = True
                stack, cells = [(y, x)], []
                while stack:
                    cy, cx = stack.pop()
                    cells.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny][nx] and rows[ny][nx] == col:
                            seen[ny][nx] = True
                            stack.append((ny, nx))
                if len(cells) > max_size:
                    continue
                counts = {}
                for cy, cx in cells:
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < H and 0 <= nx < W:
                            n = rows[ny][nx]
                            if n >= 0 and n != col:
                                counts[n] = counts.get(n, 0) + 1
                if not counts:
                    continue
                ic = pos[col]
                best = max(counts, key=lambda n: (counts[n], -de[ic, pos[n]]))
                if de[ic, pos[best]] <= max_de:
                    for cy, cx in cells:
                        rows[cy][cx] = best
                    changed = True
        if not changed:
            break
    return np.array(rows, dtype=grid.dtype).reshape(H, W)


def add_outline(grid, pal, code, allowed, notes):
    H, W = grid.shape
    g = np.full((H + 2, W + 2), -1, dtype=grid.dtype)
    g[1:-1, 1:-1] = grid
    filled = g >= 0
    if not filled.any():
        return grid
    idx = None
    if code and code.strip():
        idx = pal.index.get(code.strip().upper())
        if idx is None:
            notes.append("Outline code %s is not in %s, used the darkest color instead." % (code.strip(), pal.name))
    if idx is None:
        cand = np.asarray(allowed) if len(allowed) else np.arange(len(pal))
        idx = int(cand[np.argmin(pal.lab[cand, 0])])
    near = np.zeros_like(filled)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                near |= np.roll(np.roll(filled, dy, 0), dx, 1)   # padded, so wrap-around is empty
    g[near & ~filled] = idx
    return g


def build_pattern(img, s, pal):
    """img: RGBA photo already rotated/flipped/adjusted. Returns a Pattern."""
    t0 = time.time()
    notes = []
    iw, ih = img.size
    W, H, cs, gx0, gy0 = grid_geometry(iw, ih, s.crop, s.width, s.offset_x, s.offset_y)
    pad = int(math.ceil(cs * 2)) + 2
    padded = Image.new("RGBa", (iw + 2 * pad, ih + 2 * pad), (0, 0, 0, 0))
    padded.paste(img.convert("RGBa"), (pad, pad))
    box = (gx0 + pad, gy0 + pad, gx0 + pad + W * cs, gy0 + pad + H * cs)
    box = (max(0.0, box[0]), max(0.0, box[1]), min(float(padded.width), box[2]), min(float(padded.height), box[3]))

    cells = np.asarray(padded.resize((W, H), BOX, box=box).convert("RGBA"))
    cell_rgb = cells[..., :3].reshape(-1, 3)
    empty = (cells[..., 3] < 128)
    cell_lab = srgb_to_lab(cell_rgb).reshape(H, W, 3)
    bg_lab = None
    tol = float(s.bg_tolerance)
    if s.remove_bg:
        bg_mask, bg_lab = detect_background(cell_lab, empty, tol)
        empty = empty | bg_mask
    allowed = allowed_indices(pal, s, notes)
    grid = np.full(H * W, -1, dtype=np.int32)

    if (~empty).any():
        if s.method == "dominant":
            n = W * H
            k = 4 if n <= 6000 else (3 if n <= 16000 else 2)
            sub = np.asarray(padded.resize((W * k, H * k), BOX, box=box).convert("RGBA"))
            sub_rgb = sub[..., :3].reshape(-1, 3)
            cell_of = ((np.arange(H * k) // k)[:, None] * W + (np.arange(W * k) // k)[None, :]).reshape(-1)
            uq, inv, _ = unique_colors(sub_rgb)
            uq_lab = srgb_to_lab(uq)
            if bg_lab is not None:
                # edge cells that are mostly background get removed too (no pale halo ring)
                near_bg = (delta_e(uq_lab, bg_lab[None, :])[:, 0] <= 2.0 * tol)[inv]
                frac = np.bincount(cell_of, weights=near_bg.astype(np.float64), minlength=H * W) / float(k * k)
                empty = peel_halo(empty, frac.reshape(H, W) >= 0.5)
            empty_flat = empty.reshape(-1)
            valid = (sub[..., 3].reshape(-1) >= 128) & ~empty_flat[cell_of]
            if not valid.any():
                valid = ~empty_flat[cell_of]
            weights = np.bincount(inv[valid], minlength=len(uq)).astype(np.float64)
            keep = weights > 0
            active = select_colors(uq_lab[keep], weights[keep], pal.lab, allowed, s.max_colors)
            A = len(active)
            near = delta_e(uq_lab, pal.lab[active]).argmin(1)[inv]
            near = np.where(valid, near, A)
            votes = np.bincount(cell_of * (A + 1) + near, minlength=H * W * (A + 1))
            votes = votes.reshape(H * W, A + 1)[:, :A].astype(np.float64)
            # ties go to the color closest to the cell's average
            score = votes - 1e-3 * delta_e(cell_lab.reshape(-1, 3), pal.lab[active])
            wloc = score.argmax(1)
            thr = {1: 0.38, 2: 0.26, 3: 0.16}.get(int(s.detail), 0.0)
            if thr and A > 1:
                # Thin dark features (eyes, brows, lips, outlines) rarely fill half a bead, so a plain
                # majority vote erases them. Let a clearly darker minority win when it covers enough.
                total = votes.sum(1, keepdims=True)
                share = votes / np.maximum(total, 1.0)
                L = pal.lab[active, 0]
                contrast = delta_e(pal.lab[active], pal.lab[active])
                cand = (share >= thr) & (L[None, :] <= L[wloc][:, None] - 12.0) & (contrast[wloc] >= 20.0)
                cand_share = np.where(cand, share, -1.0)
                use = cand_share.max(1) > 0
                wloc = np.where(use, cand_share.argmax(1), wloc)
            winner = active[wloc]
            ok = (votes.max(1) > 0) & ~empty_flat
            grid = np.where(ok, winner, -1).astype(np.int32)
        else:
            if bg_lab is not None:
                d_bg = delta_e(cell_lab.reshape(-1, 3), bg_lab[None, :]).reshape(H, W)
                empty = peel_halo(empty, d_bg <= 2.2 * tol)
            sel = ~empty.reshape(-1)
            uq, inv, cnt = unique_colors(cell_rgb[sel])
            uq_lab = srgb_to_lab(uq)
            active = select_colors(uq_lab, cnt.astype(np.float64), pal.lab, allowed, s.max_colors)
            grid[sel] = active[delta_e(uq_lab, pal.lab[active]).argmin(1)][inv]
    grid = grid.reshape(H, W)

    if s.cleanup:
        grid = cleanup_grid(grid, pal, s.cleanup)
        if empty.any():
            grid, removed = remove_small_islands(grid, max_size=1 + int(s.cleanup))
            if removed:
                notes.append("Removed %d stray bead(s) floating away from the main piece." % removed)
    if s.outline:
        grid = add_outline(grid, pal, s.outline_code, allowed, notes)
    return Pattern(grid=grid, pal=pal, pitch_mm=float(s.pitch_mm), board=max(2, int(s.board)),
                   notes=notes, elapsed=time.time() - t0)


# =============================================================================
# Raster rendering (previews and PNG exports)
# =============================================================================
_FONT_PATH = None
_FONT_CACHE = {}
_FONT_CANDIDATES = [
    "DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "Arial-BoldMT.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]


def get_font(px):
    global _FONT_PATH
    px = max(6, int(px))
    if px in _FONT_CACHE:
        return _FONT_CACHE[px]
    font = None
    if _FONT_PATH is None:
        for cand in _FONT_CANDIDATES:
            try:
                font = ImageFont.truetype(cand, px)
                _FONT_PATH = cand
                break
            except Exception:
                continue
        if font is None:
            _FONT_PATH = ""
    elif _FONT_PATH:
        try:
            font = ImageFont.truetype(_FONT_PATH, px)
        except Exception:
            font = None
    if font is None:
        try:
            font = ImageFont.load_default(px)
        except TypeError:
            font = ImageFont.load_default()
    _FONT_CACHE[px] = font
    return font


def text_width(font, text):
    try:
        return font.getlength(text)
    except Exception:
        try:
            l, t, r, b = font.getbbox(text)
            return r - l
        except Exception:
            return 6 * len(text)


def draw_text_center(draw, cx, cy, text, font, fill):
    try:
        draw.text((cx, cy), text, font=font, fill=fill, anchor="mm")
    except Exception:
        try:
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
        except Exception:
            l, t, r, b = 0, 0, 6 * len(text), 10
        draw.text((cx - (r - l) / 2 - l, cy - (b - t) / 2 - t), text, font=font, fill=fill)


def grid_rgb(pat, highlight=None, empty_rgb=EMPTY_RGB):
    table = np.vstack([pat.pal.rgb.astype(np.float32), np.array([empty_rgb], np.float32)])
    g = pat.grid
    idx = np.where(g < 0, len(pat.pal), g)
    rgb = table[idx]
    if highlight is not None:
        dim = (g >= 0) & (g != highlight)
        rgb[dim] = rgb[dim] * 0.22 + 255.0 * 0.78
    return rgb


def _line_level(k, board):
    if board and k % board == 0:
        return 3
    if k % 10 == 0:
        return 2
    if k % 5 == 0:
        return 1
    return 0


_LINE_STYLE = {0: ((206, 206, 206), 1), 1: ((140, 140, 140), 1), 2: ((50, 50, 50), 2), 3: ((214, 40, 40), 2)}


def render_flat(pat, cell, lines=True, boards=True, highlight=None, labels=None, empty_rgb=EMPTY_RGB):
    g = pat.grid
    H, W = g.shape
    rgb = grid_rgb(pat, highlight, empty_rgb).astype(np.uint8)
    out = np.full((H * cell + 1, W * cell + 1, 3), 255, np.uint8)
    out[:H * cell, :W * cell] = np.repeat(np.repeat(rgb, cell, axis=0), cell, axis=1)
    board = pat.board if boards else 0
    if lines and cell >= 4:
        levels = (0, 1, 2, 3)
    elif board:
        levels = (3,)          # grid off: keep only the red board edges
    else:
        levels = ()
    if levels:
        for level in levels:
            color, thick = _LINE_STYLE[level]
            if cell < 10:
                thick = 1
            for k in range(H + 1):
                if _line_level(k, board) == level:
                    y = min(k * cell, H * cell + 1 - thick)
                    out[y:y + thick, :] = color
            for k in range(W + 1):
                if _line_level(k, board) == level:
                    x = min(k * cell, W * cell + 1 - thick)
                    out[:, x:x + thick] = color
    img = Image.fromarray(out)
    if labels and cell >= 11:
        draw = ImageDraw.Draw(img)
        fonts = {}
        for (y, x) in zip(*np.nonzero(g >= 0)):
            v = int(g[y, x])
            lab = labels.get(v)
            if not lab:
                continue
            if lab not in fonts:
                ref = get_font(100)
                px = min(cell * 0.5, cell * 0.84 * 100.0 / max(1.0, text_width(ref, lab)))
                fonts[lab] = get_font(px) if px >= 6 else None
            f = fonts[lab]
            if f is None:
                continue
            col = rgb[y, x]
            draw_text_center(draw, (x + 0.5) * cell, (y + 0.5) * cell + 0.5, lab, f, ink_for(col))
    return img


_BEAD_MASKS = {}


def _bead_masks(cell):
    if cell not in _BEAD_MASKS:
        yy, xx = np.mgrid[0:cell, 0:cell].astype(np.float32)
        c = (cell - 1) / 2.0
        dx, dy = (xx - c) / (cell / 2.0), (yy - c) / (cell / 2.0)
        r = np.sqrt(dx * dx + dy * dy)
        body = (r <= 0.94) & (r >= 0.33)
        shade = 1.0 + 0.20 * (-(dx + dy) / 2.0)
        shade = np.where(r > 0.80, shade * 0.86, shade).astype(np.float32)
        hole = r < 0.33
        peg = r <= 0.20
        _BEAD_MASKS[cell] = (body, shade, hole, peg)
    return _BEAD_MASKS[cell]


def render_beads(pat, cell, boards=True, highlight=None):
    g = pat.grid
    H, W = g.shape
    cell = max(6, int(cell))
    body, shade, hole, peg = _bead_masks(cell)
    rgb = grid_rgb(pat, highlight)
    big = np.repeat(np.repeat(rgb, cell, axis=0), cell, axis=1)
    filled = np.repeat(np.repeat(g >= 0, cell, axis=0), cell, axis=1)
    bodyT, shadeT = np.tile(body, (H, W)), np.tile(shade, (H, W))
    holeT, pegT = np.tile(hole, (H, W)), np.tile(peg, (H, W))
    out = np.empty((H * cell, W * cell, 3), np.float32)
    out[:] = BOARD_RGB
    m = bodyT & filled
    out[m] = big[m] * shadeT[m][:, None]
    out[holeT & filled] = HOLE_RGB
    out[pegT & ~filled] = PEG_RGB
    out = np.clip(out, 0, 255).astype(np.uint8)
    if boards:
        b = pat.board
        for k in range(b, H, b):
            out[k * cell - 1:k * cell + 1, :] = (214, 40, 40)
        for k in range(b, W, b):
            out[:, k * cell - 1:k * cell + 1] = (214, 40, 40)
    return Image.fromarray(out)


def max_cell_for(pat, max_pixels=16_000_000):
    return max(2, int(math.sqrt(max_pixels / max(1, pat.W * pat.H))))


def render_preview(pat, cell, style="beads", boards=True, highlight=None, labels=None, grid=True):
    cell = max(2, min(int(cell), max_cell_for(pat)))
    if style == "beads" and cell >= 6:
        return render_beads(pat, cell, boards, highlight)
    return render_flat(pat, cell, lines=grid, boards=boards, highlight=highlight,
                       labels=labels if style == "codes" else None)


def pdf_safe(text):
    """Built-in PDF and fallback fonts are Latin only; keep titles readable."""
    t = "".join(ch if 32 <= ord(ch) < 127 else "_" for ch in str(text))
    t = re.sub(r"_+", "_", t).strip(" _")
    return t or "pattern"


def spare_for(count):
    return max(2, int(math.ceil(count * 0.10)))


def render_legend_png(pat, labels, width, sw=24):
    items = pat.used_sorted()
    f = get_font(max(10, int(sw * 0.5)))
    texts = []
    for idx, cnt in items:
        code, name = pat.pal.codes[idx], pat.pal.names[idx]
        t = code if name == code else "%s %s" % (code, name)
        texts.append("%s  x%d  (buy %d)" % (t, cnt, cnt + spare_for(cnt)))
    tw = max([text_width(f, t) for t in texts] + [100])
    col_w = int(sw + 10 + tw + 24)
    ncols = max(1, (width - 20) // col_w)
    nrows = math.ceil(len(items) / ncols) if items else 0
    head = int(sw * 1.4)
    img = Image.new("RGB", (width, head + nrows * (sw + 8) + 16), "white")
    d = ImageDraw.Draw(img)
    d.text((14, 6), "Colors: %d  |  Beads: %s  (buy counts include ~10%% spare)" % (len(items), format(pat.total(), ",")),
           font=get_font(max(11, int(sw * 0.55))), fill=(30, 30, 30))
    for n, ((idx, cnt), t) in enumerate(zip(items, texts)):
        col, row = n // nrows if nrows else 0, n % nrows if nrows else 0
        x = 14 + col * col_w
        y = head + row * (sw + 8)
        rgb = tuple(int(v) for v in pat.pal.rgb[idx])
        d.rectangle([x, y, x + sw, y + sw], fill=rgb, outline=(90, 90, 90))
        lab = labels.get(idx, "")
        ref = get_font(100)
        px = min(sw * 0.5, sw * 0.84 * 100.0 / max(1.0, text_width(ref, lab)))
        draw_text_center(d, x + sw / 2, y + sw / 2 + 0.5, lab, get_font(px), ink_for(rgb))
        try:
            d.text((x + sw + 8, y + sw / 2), t, font=f, fill=(20, 20, 20), anchor="lm")
        except Exception:
            d.text((x + sw + 8, y + 2), t, font=f, fill=(20, 20, 20))
    return img


def render_chart_png(pat, title="pattern", cell=30):
    H, W = pat.H, pat.W
    cell = int(max(14, min(cell, math.sqrt(24e6 / max(1, W * H)))))
    labels = pattern_labels(pat)
    grid_img = render_flat(pat, cell, lines=True, boards=True, labels=labels)
    margin = int(cell * 1.6)
    head = int(cell * 2.0)
    legend = render_legend_png(pat, labels, grid_img.width + 2 * margin, sw=max(20, int(cell * 0.9)))
    img = Image.new("RGB", (grid_img.width + 2 * margin, head + grid_img.height + 2 * margin + legend.height), "white")
    img.paste(grid_img, (margin, head + margin))
    d = ImageDraw.Draw(img)
    f = get_font(max(9, int(cell * 0.42)))
    gy0, gy1 = head + margin, head + margin + H * cell
    for j in range(W):
        n = j + 1
        if cell >= 22 or n % 5 == 0 or n == 1:
            cx = margin + (j + 0.5) * cell
            draw_text_center(d, cx, gy0 - margin / 2, str(n), f, (70, 70, 70))
            draw_text_center(d, cx, gy1 + margin / 2, str(n), f, (70, 70, 70))
    for i in range(H):
        n = i + 1
        if cell >= 22 or n % 5 == 0 or n == 1:
            cy = gy0 + (i + 0.5) * cell
            draw_text_center(d, margin / 2, cy, str(n), f, (70, 70, 70))
            draw_text_center(d, margin + W * cell + margin / 2, cy, str(n), f, (70, 70, 70))
    bw, bh = pat.boards()
    line = "%s  |  %d x %d beads  |  %s  |  %s  |  %s, %.1f mm beads  |  boards %d x %d (%d pegs)" % (
        pdf_safe(title), W, H, format(pat.total(), ","), pat.pal.name, pat.size_text(), pat.pitch_mm,
        bw, bh, pat.board)
    d.text((margin, int(head * 0.3)), line, font=get_font(max(12, int(cell * 0.55))), fill=(20, 20, 20))
    img.paste(legend, (0, head + grid_img.height + 2 * margin))
    return img


def export_csv(pat, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "name", "hex", "count", "buy (with ~10% spare)"])
        total = 0
        for idx, cnt in pat.used_sorted():
            w.writerow([pat.pal.codes[idx], pat.pal.names[idx], "#" + pat.pal.hex[idx], cnt, cnt + spare_for(cnt)])
            total += cnt
        w.writerow(["TOTAL", "", "", total, ""])


# =============================================================================
# PDF export (vector, via reportlab)
# =============================================================================
def reportlab_ok():
    try:
        import reportlab  # noqa: F401
        return True
    except Exception:
        return False


def _split_range(a, b, n):
    step = (b - a) / float(n)
    edges = [a + int(round(step * i)) for i in range(n)] + [b]
    return list(zip(edges[:-1], edges[1:]))


def plan_tiles(pat, max_rows, max_cols, whole=False):
    H, W, board = pat.H, pat.W, pat.board
    if whole:
        return [dict(r0=0, r1=H, c0=0, c1=W, br=None, bc=None, part=None)]
    tiles = []
    for br in range(math.ceil(H / board)):
        for bc in range(math.ceil(W / board)):
            r0, r1 = br * board, min(H, (br + 1) * board)
            c0, c1 = bc * board, min(W, (bc + 1) * board)
            nr = max(1, math.ceil((r1 - r0) / float(max(1, max_rows))))
            nc = max(1, math.ceil((c1 - c0) / float(max(1, max_cols))))
            rs, cs = _split_range(r0, r1, nr), _split_range(c0, c1, nc)
            for i, (a, b) in enumerate(rs):
                for j, (c0_, c1_) in enumerate(cs):
                    if not (pat.grid[a:b, c0_:c1_] >= 0).any():
                        continue
                    part = (i * nc + j + 1, nr * nc) if nr * nc > 1 else None
                    tiles.append(dict(r0=a, r1=b, c0=c0_, c1=c1_, br=br, bc=bc, part=part))
    return tiles


def _tile_name(t):
    if t["br"] is None:
        return "Whole pattern"
    name = "Board R%d-C%d" % (t["br"] + 1, t["bc"] + 1)
    if t["part"]:
        name += "  (part %d of %d)" % t["part"]
    return name


def _pdf_fill(c, rgb):
    c.setFillColorRGB(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def _pdf_cells(c, pat, t, x0, ytop, cell, labels, show_codes):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    g = pat.grid
    fit = {}
    for r in range(t["r0"], t["r1"]):
        y = ytop - (r - t["r0"] + 1) * cell
        row = g[r]
        for col in range(t["c0"], t["c1"]):
            v = int(row[col])
            x = x0 + (col - t["c0"]) * cell
            if v < 0:
                c.setFillColorRGB(0.93, 0.93, 0.93)
                c.rect(x, y, cell, cell, stroke=0, fill=1)
                continue
            rgb = pat.pal.rgb[v]
            _pdf_fill(c, rgb)
            c.rect(x, y, cell, cell, stroke=0, fill=1)
            if show_codes:
                lab = labels.get(v, "")
                if lab not in fit:
                    w = stringWidth(lab, "Helvetica-Bold", 10.0)
                    fit[lab] = min(cell * 0.52, cell * 0.84 * 10.0 / max(w, 1e-6))
                fs = fit[lab]
                if fs >= 2.8:
                    _pdf_fill(c, ink_for(rgb))
                    c.setFont("Helvetica-Bold", fs)
                    c.drawCentredString(x + cell / 2.0, y + cell / 2.0 - fs * 0.36, lab)


def _pdf_lines(c, pat, t, x0, ytop, cell):
    styles = {0: ((0.78, 0.78, 0.78), 0.25), 1: ((0.45, 0.45, 0.45), 0.6),
              2: ((0.08, 0.08, 0.08), 0.95), 3: ((0.84, 0.12, 0.12), 1.6)}
    if cell < 6:
        styles[0] = ((0.85, 0.85, 0.85), 0.15)
    nr, nc = t["r1"] - t["r0"], t["c1"] - t["c0"]
    ybot, xr = ytop - nr * cell, x0 + nc * cell
    for level in (0, 1, 2, 3):
        col, lw = styles[level]
        c.setStrokeColorRGB(*col)
        c.setLineWidth(lw)
        for k in range(t["c0"], t["c1"] + 1):
            if _line_level(k, pat.board) == level:
                x = x0 + (k - t["c0"]) * cell
                c.line(x, ytop, x, ybot)
        for k in range(t["r0"], t["r1"] + 1):
            if _line_level(k, pat.board) == level:
                y = ytop - (k - t["r0"]) * cell
                c.line(x0, y, xr, y)
    c.setStrokeColorRGB(0.1, 0.1, 0.1)
    c.setLineWidth(0.8)
    c.rect(x0, ybot, xr - x0, ytop - ybot, stroke=1, fill=0)


def _pdf_coords(c, t, x0, ytop, cell):
    nr, nc = t["r1"] - t["r0"], t["c1"] - t["c0"]
    ybot, xr = ytop - nr * cell, x0 + nc * cell
    fs = max(4.0, min(7.0, cell * 0.6))
    every = 1 if cell >= 9 else (5 if cell >= 3 else 10)
    c.setFillColorRGB(0.3, 0.3, 0.3)
    c.setFont("Helvetica", fs)
    for k in range(t["c0"], t["c1"]):
        n = k + 1
        if n % every == 0 or k == t["c0"] or (every == 1):
            x = x0 + (k - t["c0"] + 0.5) * cell
            c.drawCentredString(x, ytop + 2.5, str(n))
            c.drawCentredString(x, ybot - fs - 1.5, str(n))
    for k in range(t["r0"], t["r1"]):
        n = k + 1
        if n % every == 0 or k == t["r0"] or (every == 1):
            y = ytop - (k - t["r0"] + 0.5) * cell - fs * 0.35
            c.drawRightString(x0 - 2.5, y, str(n))
            c.drawString(xr + 2.5, y, str(n))


def _pdf_minimap(c, pat, t, x_right, y_top, size=72.0):
    H, W, b = pat.H, pat.W, pat.board
    s = size / float(max(H, W))
    x = x_right - W * s
    c.setLineWidth(0.4)
    for br in range(math.ceil(H / b)):
        for bc in range(math.ceil(W / b)):
            rx = x + bc * b * s
            rh = min(b, H - br * b) * s
            ry = y_top - br * b * s - rh
            rw = min(b, W - bc * b) * s
            hit = t["br"] == br and t["bc"] == bc
            c.setFillColorRGB(*((0.98, 0.80, 0.80) if hit else (0.95, 0.95, 0.95)))
            c.setStrokeColorRGB(0.45, 0.45, 0.45)
            c.rect(rx, ry, rw, rh, stroke=1, fill=1)
    c.setStrokeColorRGB(0.85, 0.1, 0.1)
    c.setLineWidth(1.0)
    c.rect(x + t["c0"] * s, y_top - t["r1"] * s, (t["c1"] - t["c0"]) * s, (t["r1"] - t["r0"]) * s, stroke=1, fill=0)


def _pdf_color_list(c, pat, items, labels, x, y_top, max_w, max_h, sw=11.0):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    fs = 7.5
    xx, yy = x, y_top
    for n, (idx, cnt) in enumerate(items):
        code = pat.pal.codes[idx]
        lab = labels.get(idx, "")
        text = ("%s x%d" % (code, cnt)) if lab == code else ("%s = %s x%d" % (lab, code, cnt))
        need = sw + 3 + stringWidth(text, "Helvetica", fs) + 10
        if xx + need > x + max_w and xx > x:
            xx = x
            yy -= sw + 4
        if yy - sw < y_top - max_h:
            c.setFillColorRGB(0.3, 0.3, 0.3)
            c.setFont("Helvetica-Oblique", fs)
            c.drawString(xx, yy - sw + 3, "+%d more colors (see legend)" % (len(items) - n))
            return
        rgb = pat.pal.rgb[idx]
        _pdf_fill(c, rgb)
        c.setStrokeColorRGB(0.4, 0.4, 0.4)
        c.setLineWidth(0.4)
        c.rect(xx, yy - sw, sw, sw, stroke=1, fill=1)
        lfs = min(sw * 0.52, sw * 0.84 * 10.0 / max(stringWidth(lab, "Helvetica-Bold", 10.0), 1e-6))
        _pdf_fill(c, ink_for(rgb))
        c.setFont("Helvetica-Bold", lfs)
        c.drawCentredString(xx + sw / 2.0, yy - sw / 2.0 - lfs * 0.36, lab)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.setFont("Helvetica", fs)
        c.drawString(xx + sw + 3, yy - sw / 2.0 - fs * 0.36, text)
        xx += need


def _pdf_legend_table(c, pat, labels, items, x, y_top, w, y_min):
    """Two-column legend. Returns how many items were drawn."""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    row_h, sw = 15.0, 12.0
    col_w = (w - 16) / 2.0
    rows_fit = int((y_top - y_min - 14) // row_h)
    if rows_fit <= 0:
        return 0
    c.setFont("Helvetica-Bold", 8)
    c.setFillColorRGB(0.2, 0.2, 0.2)
    for k in range(2):
        cx = x + k * (col_w + 16)
        c.drawString(cx + sw + 6, y_top - 9, "code / name")
        c.drawRightString(cx + col_w - 44, y_top - 9, "beads")
        c.drawRightString(cx + col_w, y_top - 9, "buy")
    n_draw = min(len(items), rows_fit * 2)
    per_col = min(rows_fit, max(1, math.ceil(n_draw / 2.0)))
    for n in range(n_draw):
        idx, cnt = items[n]
        k, r = n // per_col, n % per_col
        cx = x + k * (col_w + 16)
        cy = y_top - 14 - (r + 1) * row_h
        rgb = pat.pal.rgb[idx]
        _pdf_fill(c, rgb)
        c.setStrokeColorRGB(0.4, 0.4, 0.4)
        c.setLineWidth(0.4)
        c.rect(cx, cy + 1.5, sw, sw, stroke=1, fill=1)
        lab = labels.get(idx, "")
        lfs = min(sw * 0.52, sw * 0.84 * 10.0 / max(stringWidth(lab, "Helvetica-Bold", 10.0), 1e-6))
        _pdf_fill(c, ink_for(rgb))
        c.setFont("Helvetica-Bold", lfs)
        c.drawCentredString(cx + sw / 2.0, cy + 1.5 + sw / 2.0 - lfs * 0.36, lab)
        name = pat.pal.describe(idx)
        c.setFillColorRGB(0.08, 0.08, 0.08)
        c.setFont("Helvetica", 8.5)
        maxw = col_w - sw - 6 - 90
        while stringWidth(name, "Helvetica", 8.5) > maxw and len(name) > 4:
            name = name[:-2]
        c.drawString(cx + sw + 6, cy + 4.5, pdf_safe(name))
        c.drawRightString(cx + col_w - 44, cy + 4.5, format(cnt, ","))
        c.drawRightString(cx + col_w, cy + 4.5, format(cnt + spare_for(cnt), ","))
    return n_draw


def _summary_lines(pat):
    bw, bh = pat.boards()
    used = pat.used_sorted()
    return [
        "%d x %d beads  |  %s beads total  |  %d colors  |  palette: %s" % (
            pat.W, pat.H, format(pat.total(), ","), len(used), pat.pal.name),
        "Finished size about %s with %.1f mm beads" % (pat.size_text(), pat.pitch_mm),
        "Boards: %d across x %d down (%d x %d pegs each). Red lines in the chart = board edges." % (
            bw, bh, pat.board, pat.board),
        "Made %s with %s. Screen colors approximate real beads, so check your color card." % (
            date.today().isoformat(), APP_NAME),
    ]


def export_chart_pdf(pat, path, paper="Letter", mode=CHART_MODES[0], title="pattern"):
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as rl_canvas
    pw, ph = PAPERS.get(paper, PAPERS["Letter"])
    m = 34.0
    labels = pattern_labels(pat)
    items = pat.used_sorted()
    c = rl_canvas.Canvas(path, pagesize=(pw, ph))
    c.setTitle("%s - bead pattern" % pdf_safe(title))
    c.setAuthor(APP_NAME)

    # --- page 1: overview + legend
    c.setFont("Helvetica-Bold", 17)
    c.setFillColorRGB(0.05, 0.05, 0.05)
    c.drawString(m, ph - m - 14, "%s  -  bead pattern" % pdf_safe(title))
    c.setFont("Helvetica", 9)
    y = ph - m - 32
    for line in _summary_lines(pat):
        c.drawString(m, y, line)
        y -= 12.5
    cellpx = max(6, min(20, 1600 // max(pat.W, pat.H)))
    prev = render_beads(pat, cellpx, boards=False)
    aw, ah = pw - 2 * m, (ph - 2 * m) * 0.50
    s = min(aw / prev.width, ah / prev.height)
    dw, dh = prev.width * s, prev.height * s
    px, py = m + (aw - dw) / 2.0, y - 6 - dh
    c.drawImage(ImageReader(prev), px, py, dw, dh)
    cell = dw / pat.W
    bw, bh = pat.boards()
    if bw * bh > 1:
        c.setStrokeColorRGB(0.84, 0.12, 0.12)
        c.setLineWidth(1.2)
        for k in range(1, bw):
            c.line(px + k * pat.board * cell, py, px + k * pat.board * cell, py + dh)
        for k in range(1, bh):
            c.line(px, py + dh - k * pat.board * cell, px + dw, py + dh - k * pat.board * cell)
        for br in range(bh):
            for bc in range(bw):
                cx = px + (bc * pat.board + min(pat.board, pat.W - bc * pat.board) / 2.0) * cell
                cy = py + dh - (br * pat.board + min(pat.board, pat.H - br * pat.board) / 2.0) * cell
                tag = "R%d-C%d" % (br + 1, bc + 1)
                c.setFillColorRGB(1, 1, 1)
                c.roundRect(cx - 17, cy - 6, 34, 12, 3, stroke=0, fill=1)
                c.setFillColorRGB(0.75, 0.08, 0.08)
                c.setFont("Helvetica-Bold", 7.5)
                c.drawCentredString(cx, cy - 2.7, tag)
    y = py - 14
    drawn = _pdf_legend_table(c, pat, labels, items, m, y, pw - 2 * m, m)
    while drawn < len(items):
        c.showPage()
        c.setFont("Helvetica-Bold", 12)
        c.setFillColorRGB(0.05, 0.05, 0.05)
        c.drawString(m, ph - m - 10, "Colors (continued)")
        drawn += _pdf_legend_table(c, pat, labels, items[drawn:], m, ph - m - 22, pw - 2 * m, m)

    # --- chart pages
    whole = mode == CHART_MODES[1]
    min_cell = 15.0 if mode == CHART_MODES[2] else 8.5
    head_h, foot_h, side = 54.0, 66.0, 18.0
    gw, gh = pw - 2 * m - 2 * side, ph - 2 * m - head_h - foot_h - 2 * side
    tiles = plan_tiles(pat, int(gh // min_cell), int(gw // min_cell), whole=whole)
    for t in tiles:
        c.showPage()
        nr, nc = t["r1"] - t["r0"], t["c1"] - t["c0"]
        cell = min(gw / nc, gh / nr, 30.0)
        x0 = m + side + (gw - nc * cell) / 2.0
        ytop = ph - m - head_h - side
        c.setFillColorRGB(0.05, 0.05, 0.05)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(m, ph - m - 12, _tile_name(t))
        sub = pat.grid[t["r0"]:t["r1"], t["c0"]:t["c1"]]
        c.setFont("Helvetica", 8.5)
        c.drawString(m, ph - m - 26, "Columns %d-%d, rows %d-%d  |  %s beads on this page  |  %s" % (
            t["c0"] + 1, t["c1"], t["r0"] + 1, t["r1"], format(int((sub >= 0).sum()), ","), pdf_safe(title)))
        c.drawString(m, ph - m - 38, "Lines: thin = 1 bead, grey = every 5, black = every 10, red = board edge. "
                                     "Grey cells = no bead.")
        _pdf_minimap(c, pat, t, pw - m, ph - m - 2, size=46.0)
        _pdf_cells(c, pat, t, x0, ytop, cell, labels, show_codes=cell >= 5.5)
        _pdf_lines(c, pat, t, x0, ytop, cell)
        _pdf_coords(c, t, x0, ytop, cell)
        tile_items = pat.used_sorted(sub)
        _pdf_color_list(c, pat, tile_items, labels, m, m + foot_h - 6, pw - 2 * m, foot_h - 6)
    c.save()


def _pdf_ruler(c, x, y, length_mm):
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.6)
    c.line(x, y, x + length_mm * MM, y)
    c.setFont("Helvetica", 6)
    c.setFillColorRGB(0, 0, 0)
    for k in range(int(length_mm) + 1):
        h = 7 if k % 10 == 0 else (4.5 if k % 5 == 0 else 2.5)
        c.line(x + k * MM, y, x + k * MM, y + h)
        if k % 10 == 0:
            c.drawCentredString(x + k * MM, y + 9, "%d" % (k // 10))
    c.drawString(x + length_mm * MM + 6, y + 1, "cm")


def export_true_scale_pdf(pat, path, paper="Letter", title="pattern"):
    from reportlab.pdfgen import canvas as rl_canvas
    pw, ph = PAPERS.get(paper, PAPERS["Letter"])
    m = 28.0
    labels = pattern_labels(pat)
    pitch = pat.pitch_mm * MM
    head_h, ruler_h, foot_h, side = 40.0, 26.0, 60.0, 16.0
    gw = pw - 2 * m - 2 * side
    gh = ph - 2 * m - head_h - ruler_h - foot_h - 2 * side
    tiles = plan_tiles(pat, int(gh // pitch), int(gw // pitch))
    c = rl_canvas.Canvas(path, pagesize=(pw, ph))
    c.setTitle("%s - true size pattern" % pdf_safe(title))
    c.setAuthor(APP_NAME)
    ruler_mm = 100 if gw >= 100 * MM + 40 else 50
    for t in tiles:
        nr, nc = t["r1"] - t["r0"], t["c1"] - t["c0"]
        x0 = m + side
        ytop = ph - m - head_h - ruler_h - side
        c.setFillColorRGB(0.05, 0.05, 0.05)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(m, ph - m - 11, "%s  -  TRUE SIZE (%.1f mm beads)" % (_tile_name(t), pat.pitch_mm))
        c.setFont("Helvetica", 8)
        c.drawString(m, ph - m - 24, "Print at 100%% / Actual size (turn OFF \"Fit to page\"). The ruler must measure "
                                     "exactly %d cm. Then lay a clear pegboard on top." % (ruler_mm // 10))
        c.drawString(m, ph - m - 35, "Columns %d-%d, rows %d-%d of %d x %d  |  %s" % (
            t["c0"] + 1, t["c1"], t["r0"] + 1, t["r1"], pat.W, pat.H, pdf_safe(title)))
        _pdf_ruler(c, m, ph - m - head_h - 14, ruler_mm)
        _pdf_minimap(c, pat, t, pw - m, ph - m - 2, size=40.0)
        _pdf_cells(c, pat, t, x0, ytop, pitch, labels, show_codes=pitch >= 12.0)
        _pdf_lines(c, pat, t, x0, ytop, pitch)
        _pdf_coords(c, t, x0, ytop, pitch)
        sub = pat.grid[t["r0"]:t["r1"], t["c0"]:t["c1"]]
        _pdf_color_list(c, pat, pat.used_sorted(sub), labels, m, m + foot_h - 4, pw - 2 * m, foot_h - 4)
        c.showPage()
    c.save()


def export_all(pat, base, paper="Letter", mode=CHART_MODES[0], title="pattern"):
    if pat.total() == 0:
        raise ValueError("The pattern has no beads (everything was removed as background).")
    folder = os.path.dirname(os.path.abspath(base))
    os.makedirs(folder, exist_ok=True)
    files, warnings = [], []
    p = base + "_chart.png"
    render_chart_png(pat, title).save(p, optimize=True)
    files.append(p)
    p = base + "_preview.png"
    render_beads(pat, max(6, min(24, 1800 // max(pat.W, pat.H)))).save(p, optimize=True)
    files.append(p)
    p = base + "_colors.csv"
    export_csv(pat, p)
    files.append(p)
    if reportlab_ok():
        p = base + "_chart.pdf"
        export_chart_pdf(pat, p, paper, mode, title)
        files.append(p)
        p = base + "_true_size.pdf"
        export_true_scale_pdf(pat, p, paper, title)
        files.append(p)
    else:
        warnings.append("PDFs skipped: run  pip install reportlab  to enable them.")
    return files, warnings


def open_folder(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


TIPS = """WHAT MAKES A GOOD 拼豆 PATTERN

Photo
  - One clear subject, plain background, even light. Busy scenes turn into noise.
  - Crop tight. Everything outside the crop costs beads and detail.
  - Faces: click "Level eyes" first, then turn on the bead grid and nudge it so
    each eye sits inside a bead instead of splitting across two. Lopsided eyes
    are the #1 reason bead faces look tilted.

Size (detail vs. size)
  - More beads = more detail but a bigger piece and more work.
  - Rough guide: 20-30 wide = icon/keychain, 40-60 = faces start to read,
    80+ = photo-like. Mini 2.6 mm beads give about twice the detail per cm of 5 mm.
  - Common Chinese mini boards are 52 x 52 pegs (about 13.5 cm). Perler/Hama
    5 mm boards are 29 x 29 (about 14.5 cm). Set yours under "Board pegs".
    Red lines in the chart are board edges.

Colors
  - 8-20 colors is the sweet spot. Each extra color is another bag to sort.
  - "Clean edges" (dominant color) avoids the grey fringe that averaging creates.
    "Smooth shading" (average) can suit soft photos and small dark details.
  - "Face details" (0-3) keeps eyes, brows and lips that a plain majority vote
    would erase. 2 is a good default; try 3 for small patterns.
  - Fill in "Only use codes" with the colors you actually own.
  - Screen colors are approximations. Hold your physical color card next to it.

Cleanup and strength
  - Light cleanup merges random single beads that are just photo noise.
    Fix eyes and highlights by hand afterwards (Paint tool, Mirror on).
  - Beads that touch only at a corner don't fuse well and can snap after
    ironing. Red circles mark those spots; fill a bead next to them.
  - "Add outline" gives a crisp edge and also ties loose bits together.

Printing
  - *_true_size.pdf: print at 100% (Actual size), check the ruler, slide it
    under a clear pegboard and place beads right on top.
  - *_chart.pdf: readable color codes, one page per board (or Large print).
  - *_chart.png: the whole pattern as one image for your phone or iPad.
  - *_colors.csv: shopping list with ~10% spare.
"""

ABOUT = """%s - %s   (version %s)

Privacy: this app has no networking code at all. Your photos are read from disk,
processed in memory on this computer, and only written where you export.

Bead palettes: bundled from the open datasets HansBug/pindou-color-data and
maxcleme/beadcolors (packaged by the Apache-2.0 project HansBug/pypindou).
Color matching uses CIEDE2000, the perceptual color-difference formula.
""" % (APP_NAME, APP_SUBTITLE, APP_VERSION)


# =============================================================================
# Desktop app (Tkinter)
# =============================================================================
class PindouApp:
    def __init__(self, root, initial_path=None):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from PIL import ImageTk
        self.tk, self.ttk, self.fd, self.mb, self.ImageTk = tk, ttk, filedialog, messagebox, ImageTk
        self.root = root
        self.pals = palettes()
        self.label_to_key = {p.label: k for k, p in self.pals.items()}
        self.src = None
        self.src_preview = None
        self.src_path = None
        self.disp_geo = None
        self.disp_key = None
        self.disp_size = None
        self.crop = (0.0, 0.0, 1.0, 1.0)
        self.pattern = None
        self.base_grid = None
        self.edits = {}
        self.undo = []
        self.edit_sig = None
        self.cur_color = None
        self.level_mode = False
        self.level_pts = []
        self.last_settings = None
        self.weak = []
        self._geo_cache = None
        self._geo_lock = threading.Lock()
        self._job = None
        self._busy = False
        self._pending = False
        self._token = 0
        self._q = queue.Queue()
        self._drag = None
        self._stroke = None
        self._photo_tk = None
        self._photo_base_key = None
        self._photo_xf = None
        self._pattern_tk = None
        self._render_cell = 10
        self._last_shape = None
        self._last_rot = 0.0
        self._last_flip = False
        self._suspend = False
        self._build_vars()
        self._build_ui()
        self._bind_keys()
        self._on_palette_change(init=True)
        root.after(60, self._poll)
        if initial_path:
            root.after(250, lambda: self.open_photo(initial_path))

    # ------------------------------------------------------------------ state
    def _build_vars(self):
        tk = self.tk
        self.palette_var = tk.StringVar(value=self.pals["mard221"].label)
        self.width_var = tk.IntVar(value=52)
        self.pitch_var = tk.StringVar(value="2.6")
        self.board_var = tk.StringVar(value="52")
        self.beadtype_var = tk.StringVar(value=BEAD_TYPES[0][0])
        self.maxcol_var = tk.IntVar(value=16)
        self.method_var = tk.StringVar(value="dominant")
        self.detail_var = tk.IntVar(value=2)
        self.special_var = tk.BooleanVar(value=False)
        self.only_var = tk.StringVar()
        self.excl_var = tk.StringVar()
        self.rot_var = tk.DoubleVar(value=0.0)
        self.flip_var = tk.BooleanVar(value=False)
        self.offx_var = tk.DoubleVar(value=0.0)
        self.offy_var = tk.DoubleVar(value=0.0)
        self.bright_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.sat_var = tk.DoubleVar(value=1.0)
        self.bg_var = tk.BooleanVar(value=False)
        self.bgtol_var = tk.DoubleVar(value=12.0)
        self.clean_var = tk.IntVar(value=1)
        self.outline_var = tk.BooleanVar(value=False)
        self.outline_code_var = tk.StringVar()
        self.aspect_var = tk.StringVar(value="Free")
        self.showgrid_var = tk.BooleanVar(value=True)
        self.style_var = tk.StringVar(value="beads")
        self.zoom_var = tk.IntVar(value=10)
        self.boards_var = tk.BooleanVar(value=True)
        self.gridlines_var = tk.BooleanVar(value=True)
        self.weak_var = tk.BooleanVar(value=True)
        self.hilite_var = tk.BooleanVar(value=False)
        self.tool_var = tk.StringVar(value="paint")
        self.mirror_var = tk.BooleanVar(value=False)
        self.paper_var = tk.StringVar(value="Letter")
        self.mode_var = tk.StringVar(value=CHART_MODES[0])
        self.target_cm_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Open a photo to start. It never leaves this computer.")
        self.size_var = tk.StringVar(value="")
        self.clean_name_var = tk.StringVar(value=CLEAN_NAMES[1])
        self.paint_combo_var = tk.StringVar()

        for v in (self.width_var, self.maxcol_var, self.method_var, self.detail_var, self.special_var, self.only_var,
                  self.excl_var, self.offx_var, self.offy_var, self.bright_var, self.contrast_var,
                  self.sat_var, self.bg_var, self.bgtol_var, self.clean_var, self.outline_var,
                  self.outline_code_var, self.pitch_var, self.board_var):
            v.trace_add("write", self._on_change)
        self.rot_var.trace_add("write", self._on_rot_change)
        self.flip_var.trace_add("write", self._on_flip_change)
        self.palette_var.trace_add("write", lambda *_: self._on_palette_change())
        self.beadtype_var.trace_add("write", self._on_beadtype)
        self.aspect_var.trace_add("write", self._on_aspect)
        self.showgrid_var.trace_add("write", lambda *_: self._draw_photo_overlay())
        for v in (self.style_var, self.boards_var, self.gridlines_var, self.weak_var, self.hilite_var, self.zoom_var):
            v.trace_add("write", lambda *_: self.refresh_pattern())
        self.clean_var.trace_add("write", lambda *_: self.clean_name_var.set(
            CLEAN_NAMES.get(self._int(self.clean_var, 1), "")))

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        tk, ttk, root = self.tk, self.ttk, self.root
        root.title("%s - %s" % (APP_NAME, APP_SUBTITLE))
        root.geometry("1380x880")
        root.minsize(1020, 660)
        style = ttk.Style()
        if sys.platform.startswith("linux"):
            try:
                style.theme_use("clam")
            except Exception:
                pass
        bg = style.lookup("TFrame", "background") or "#ececec"
        import tkinter.font as tkfont
        base = tkfont.nametofont("TkDefaultFont")
        self.font_title = base.copy()
        self.font_title.configure(size=15, weight="bold")
        self.font_sub = base.copy()
        self.font_sub.configure(size=12, weight="normal")
        fams = set(tkfont.families())
        for fam in ("PingFang SC", "Hiragino Sans GB", "Microsoft YaHei UI", "Microsoft YaHei",
                    "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei", "SimHei"):
            if fam in fams:   # a font that surely has the Chinese subtitle glyphs
                self.font_sub.configure(family=fam)
                break
        self.font_bold = base.copy()
        self.font_bold.configure(size=11, weight="bold")

        status = ttk.Label(root, textvariable=self.status_var, anchor="w", padding=(10, 4))
        status.pack(side="bottom", fill="x")
        outer = ttk.Frame(root)
        outer.pack(fill="both", expand=True)

        # left: scrollable controls
        left = ttk.Frame(outer)
        left.pack(side="left", fill="y")
        self.lcanvas = tk.Canvas(left, width=400, highlightthickness=0, bd=0, background=bg)
        lsb = ttk.Scrollbar(left, orient="vertical", command=self.lcanvas.yview)
        self.lcanvas.configure(yscrollcommand=lsb.set, yscrollincrement=16)
        lsb.pack(side="right", fill="y")
        self.lcanvas.pack(side="left", fill="y")
        panel = ttk.Frame(self.lcanvas)
        win = self.lcanvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda e: self.lcanvas.configure(scrollregion=self.lcanvas.bbox("all")))
        self.lcanvas.bind("<Configure>", lambda e: self.lcanvas.itemconfigure(win, width=e.width))
        self._build_controls(panel)

        # right: colors used
        right = ttk.Frame(outer, padding=(6, 6))
        right.pack(side="right", fill="y")
        self._build_legend(right)

        # center: tabs
        self.nb = ttk.Notebook(outer)
        self.nb.pack(side="left", fill="both", expand=True, padx=2, pady=4)
        self.photo_tab = ttk.Frame(self.nb)
        self.pattern_tab = ttk.Frame(self.nb)
        self.nb.add(self.photo_tab, text="  1 · Photo, level & crop  ")
        self.nb.add(self.pattern_tab, text="  2 · Pattern & edit  ")
        self._build_photo_tab()
        self._build_pattern_tab()
        root.bind_all("<MouseWheel>", self._on_wheel, add="+")
        root.bind_all("<Button-4>", self._on_wheel, add="+")
        root.bind_all("<Button-5>", self._on_wheel, add="+")

    def _section(self, parent, title):
        f = self.ttk.LabelFrame(parent, text=title, padding=(8, 4, 8, 6))
        f.pack(fill="x", padx=(8, 4), pady=4)
        return f

    def _slider(self, parent, label, var, frm, to, res, length=180):
        tk, ttk = self.tk, self.ttk
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Label(row, text=label, width=12).pack(side="left", anchor="s", pady=(0, 3))
        s = tk.Scale(row, variable=var, from_=frm, to=to, resolution=res, orient="horizontal",
                     length=length, highlightthickness=0, bd=0, sliderlength=18, width=12)
        s.pack(side="left", fill="x", expand=True)
        return s

    def _build_controls(self, p):
        tk, ttk = self.tk, self.ttk
        head = ttk.Frame(p, padding=(10, 8, 6, 2))
        head.pack(fill="x")
        ttk.Label(head, text=APP_NAME, font=self.font_title).pack(anchor="w")
        ttk.Label(head, text=APP_SUBTITLE, font=self.font_sub, foreground="#b4235a").pack(anchor="w")
        # Photo
        f = self._section(p, "Photo")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(0, 2))
        ttk.Button(row, text="Open photo...", command=self.open_photo).pack(side="left")
        ttk.Checkbutton(row, text="Mirror", variable=self.flip_var).pack(side="left", padx=8)
        self.file_lbl = ttk.Label(f, text="(no photo yet)", foreground="#666")
        self.file_lbl.pack(anchor="w")
        self._slider(f, "Rotate (°)", self.rot_var, -45, 45, 0.1)
        row = ttk.Frame(f)
        row.pack(fill="x", pady=2)
        ttk.Button(row, text="Level eyes", command=self.start_level).pack(side="left")
        ttk.Button(row, text="-0.5°", width=5, command=lambda: self._nudge_rot(-0.5)).pack(side="left", padx=2)
        ttk.Button(row, text="+0.5°", width=5, command=lambda: self._nudge_rot(0.5)).pack(side="left")
        ttk.Button(row, text="Reset", width=6, command=lambda: self.rot_var.set(0.0)).pack(side="left", padx=2)
        row = ttk.Frame(f)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Crop shape").pack(side="left")
        ttk.Combobox(row, textvariable=self.aspect_var, values=list(ASPECTS), width=13,
                     state="readonly").pack(side="left", padx=6)
        ttk.Button(row, text="Full photo", command=self.reset_crop).pack(side="left")
        ttk.Checkbutton(f, text="Show bead grid + center line on photo",
                        variable=self.showgrid_var).pack(anchor="w")
        ttk.Label(f, text="Nudge the grid so eyes sit inside beads:", foreground="#555").pack(anchor="w", pady=(4, 0))
        self._slider(f, "Grid nudge X", self.offx_var, -0.5, 0.5, 0.05)
        self._slider(f, "Grid nudge Y", self.offy_var, -0.5, 0.5, 0.05)

        # Size
        f = self._section(p, "Size  (more beads = more detail, bigger piece)")
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Label(row, text="Bead type").pack(side="left")
        ttk.Combobox(row, textvariable=self.beadtype_var, values=[b[0] for b in BEAD_TYPES], width=30,
                     state="readonly").pack(side="left", padx=4)
        row = ttk.Frame(f)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Peg pitch mm").pack(side="left")
        ttk.Spinbox(row, from_=1.0, to=20.0, increment=0.1, textvariable=self.pitch_var, width=5).pack(side="left", padx=4)
        ttk.Label(row, text="  Board pegs").pack(side="left")
        ttk.Combobox(row, textvariable=self.board_var, values=["29", "52", "57", "58", "48"], width=5).pack(side="left", padx=4)
        self._slider(f, "Width (beads)", self.width_var, 8, 240, 1)
        row = ttk.Frame(f)
        row.pack(fill="x", pady=2)
        for text, fn in (("Keychain", lambda: 20), ("1 board", lambda: self._board()),
                         ("2 boards", lambda: 2 * self._board()), ("3 boards", lambda: 3 * self._board())):
            ttk.Button(row, text=text, width=-8,
                       command=lambda fn=fn: self.width_var.set(int(fn()))).pack(side="left", padx=1)
        row = ttk.Frame(f)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Target width").pack(side="left")
        ttk.Entry(row, textvariable=self.target_cm_var, width=6).pack(side="left", padx=4)
        ttk.Label(row, text="cm").pack(side="left")
        ttk.Button(row, text="Set", width=5, command=self._set_target_cm).pack(side="left", padx=4)
        ttk.Label(f, textvariable=self.size_var, justify="left", foreground="#1f4e8c").pack(anchor="w", pady=(4, 0))

        # Colors
        f = self._section(p, "Colors")
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Label(row, text="Brand").pack(side="left")
        ttk.Combobox(row, textvariable=self.palette_var, values=[pp.label for pp in self.pals.values()],
                     width=33, state="readonly").pack(side="left", padx=4)
        self._slider(f, "Max colors", self.maxcol_var, 2, 60, 1)
        ttk.Radiobutton(f, text="Clean edges (dominant color, best default)", value="dominant",
                        variable=self.method_var).pack(anchor="w")
        ttk.Radiobutton(f, text="Smooth shading (average color)", value="average",
                        variable=self.method_var).pack(anchor="w")
        self._slider(f, "Face details", self.detail_var, 0, 3, 1)
        ttk.Label(f, text="stops eyes, brows and lips from vanishing",
                  foreground="#666").pack(anchor="e")
        ttk.Checkbutton(f, text="Allow special beads (pearl, glow, clear, neon...)",
                        variable=self.special_var).pack(anchor="w", pady=(2, 0))
        ttk.Label(f, text="Only use these codes (the colors you own):").pack(anchor="w", pady=(4, 0))
        ttk.Entry(f, textvariable=self.only_var).pack(fill="x")
        ttk.Label(f, text="Never use these codes:").pack(anchor="w", pady=(2, 0))
        ttk.Entry(f, textvariable=self.excl_var).pack(fill="x")

        # Adjust
        f = self._section(p, "Adjust photo")
        self._slider(f, "Brightness", self.bright_var, 0.5, 1.8, 0.05)
        self._slider(f, "Contrast", self.contrast_var, 0.5, 2.0, 0.05)
        self._slider(f, "Saturation", self.sat_var, 0.0, 2.0, 0.05)
        ttk.Button(f, text="Reset adjustments", command=self._reset_adjust).pack(anchor="w", pady=2)

        # Background & cleanup
        f = self._section(p, "Background, cleanup, outline")
        ttk.Checkbutton(f, text="Remove plain background (from the edges in)", variable=self.bg_var).pack(anchor="w")
        self._slider(f, "BG tolerance", self.bgtol_var, 3, 40, 1)
        self._slider(f, "Cleanup", self.clean_var, 0, 3, 1)
        ttk.Label(f, textvariable=self.clean_name_var, foreground="#555").pack(anchor="e")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=2)
        ttk.Checkbutton(row, text="Add outline", variable=self.outline_var).pack(side="left")
        ttk.Label(row, text="  color code").pack(side="left")
        ttk.Entry(row, textvariable=self.outline_code_var, width=7).pack(side="left", padx=4)
        ttk.Label(f, text="(blank = darkest color)", foreground="#666").pack(anchor="e")

        # Edit
        f = self._section(p, "Edit beads (on the Pattern tab)")
        row = ttk.Frame(f)
        row.pack(fill="x")
        for text, val in (("Paint", "paint"), ("Erase", "erase"), ("Fill", "fill"), ("Pick", "pick")):
            ttk.Radiobutton(row, text=text, value=val, variable=self.tool_var).pack(side="left", padx=(0, 6))
        ttk.Checkbutton(f, text="Mirror edits left <-> right (symmetric faces)",
                        variable=self.mirror_var).pack(anchor="w", pady=2)
        row = ttk.Frame(f)
        row.pack(fill="x", pady=2)
        self.swatch = tk.Label(row, text="  none  ", width=8, relief="groove", bg="#ffffff")
        self.swatch.pack(side="left")
        self.paint_combo = ttk.Combobox(row, textvariable=self.paint_combo_var, width=22, state="readonly")
        self.paint_combo.pack(side="left", padx=4)
        self.paint_combo.bind("<<ComboboxSelected>>", self._on_paint_combo)
        row = ttk.Frame(f)
        row.pack(fill="x", pady=2)
        ttk.Button(row, text="Undo", command=self.undo_edit).pack(side="left")
        ttk.Button(row, text="Clear all edits", command=self.clear_edits).pack(side="left", padx=4)
        ttk.Label(f, text="Right-click erases. Click a color in the list to paint with it.",
                  foreground="#666", wraplength=310).pack(anchor="w")

        # Export
        f = self._section(p, "Print / export")
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Label(row, text="Paper").pack(side="left")
        ttk.Combobox(row, textvariable=self.paper_var, values=list(PAPERS), width=7,
                     state="readonly").pack(side="left", padx=4)
        ttk.Label(row, text="Chart").pack(side="left", padx=(6, 0))
        ttk.Combobox(row, textvariable=self.mode_var, values=CHART_MODES, width=22,
                     state="readonly").pack(side="left", padx=4)
        row = ttk.Frame(f)
        row.pack(fill="x", pady=4)
        self.export_btn = ttk.Button(row, text="Export pattern...", command=self.export)
        self.export_btn.pack(side="left")
        ttk.Button(row, text="Tips", command=lambda: self._text_window("Tips", TIPS)).pack(side="left", padx=4)
        ttk.Button(row, text="About & privacy", command=lambda: self._text_window("About", ABOUT)).pack(side="left")
        ttk.Frame(p, height=12).pack()

    def _build_legend(self, right):
        tk, ttk = self.tk, self.ttk
        ttk.Label(right, text="Colors used", font=self.font_bold).pack(anchor="w")
        fr = ttk.Frame(right)
        fr.pack(fill="both", expand=True, pady=4)
        self.legend = tk.Listbox(fr, width=30, font="TkFixedFont", activestyle="none",
                                 exportselection=False, selectmode="browse")
        sb = ttk.Scrollbar(fr, command=self.legend.yview)
        self.legend.configure(yscrollcommand=sb.set)
        self.legend.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.legend.bind("<<ListboxSelect>>", self._on_legend_select)
        self._legend_idx = []
        ttk.Checkbutton(right, text="Highlight selected color", variable=self.hilite_var).pack(anchor="w")
        self.totals_lbl = ttk.Label(right, text="", justify="left", wraplength=250)
        self.totals_lbl.pack(anchor="w", pady=(6, 2))
        self.checks_lbl = ttk.Label(right, text="", justify="left", wraplength=250, foreground="#9a3412")
        self.checks_lbl.pack(anchor="w", pady=2)

    def _build_photo_tab(self):
        tk = self.tk
        self.photo_canvas = tk.Canvas(self.photo_tab, background="#2e2e30", highlightthickness=0, cursor="tcross")
        self.photo_canvas.pack(fill="both", expand=True)
        self.photo_canvas.bind("<Configure>", lambda e: self.refresh_photo())
        self.photo_canvas.bind("<ButtonPress-1>", self._on_photo_press)
        self.photo_canvas.bind("<B1-Motion>", self._on_photo_motion)
        self.photo_canvas.bind("<ButtonRelease-1>", self._on_photo_release)

    def _build_pattern_tab(self):
        tk, ttk = self.tk, self.ttk
        bar = ttk.Frame(self.pattern_tab, padding=(6, 4))
        bar.pack(fill="x")
        ttk.Label(bar, text="View").pack(side="left")
        for text, val in (("Beads", "beads"), ("Flat", "flat"), ("Codes", "codes")):
            ttk.Radiobutton(bar, text=text, value=val, variable=self.style_var).pack(side="left", padx=3)
        ttk.Label(bar, text="   Zoom").pack(side="left")
        tk.Scale(bar, variable=self.zoom_var, from_=2, to=40, orient="horizontal", length=120, showvalue=False,
                 highlightthickness=0, bd=0, sliderlength=16, width=10).pack(side="left")
        ttk.Button(bar, text="Fit", width=4, command=self.fit_zoom).pack(side="left", padx=4)
        bar2 = ttk.Frame(self.pattern_tab, padding=(6, 0, 6, 4))
        bar2.pack(fill="x")
        ttk.Label(bar2, text="Show").pack(side="left")
        ttk.Checkbutton(bar2, text="Grid lines", variable=self.gridlines_var).pack(side="left", padx=(6, 0))
        ttk.Checkbutton(bar2, text="Board edges (red)", variable=self.boards_var).pack(side="left", padx=10)
        ttk.Checkbutton(bar2, text="Weak spots (red circles)", variable=self.weak_var).pack(side="left")
        wrap = ttk.Frame(self.pattern_tab)
        wrap.pack(fill="both", expand=True)
        self.pat_canvas = tk.Canvas(wrap, background="#f4f4f2", highlightthickness=0, cursor="pencil",
                                    xscrollincrement=16, yscrollincrement=16)
        xs = ttk.Scrollbar(wrap, orient="horizontal", command=self.pat_canvas.xview)
        ys = ttk.Scrollbar(wrap, orient="vertical", command=self.pat_canvas.yview)
        self.pat_canvas.configure(xscrollcommand=xs.set, yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        xs.pack(side="bottom", fill="x")
        self.pat_canvas.pack(fill="both", expand=True)
        c = self.pat_canvas
        c.bind("<ButtonPress-1>", lambda e: self._on_pat_press(e, False))
        c.bind("<B1-Motion>", self._on_pat_drag)
        c.bind("<ButtonRelease-1>", self._on_pat_release)
        for b in ("<ButtonPress-3>", "<ButtonPress-2>"):
            c.bind(b, lambda e: self._on_pat_press(e, True))
        for b in ("<B3-Motion>", "<B2-Motion>"):
            c.bind(b, self._on_pat_drag)
        for b in ("<ButtonRelease-3>", "<ButtonRelease-2>"):
            c.bind(b, self._on_pat_release)
        c.bind("<Motion>", self._on_pat_hover)
        c.bind("<Configure>", lambda e: self.refresh_pattern() if self.pattern is None else None)

    def _bind_keys(self):
        r = self.root
        for mod in ("Control", "Command"):
            for key, fn in (("z", self.undo_edit), ("o", self.open_photo), ("e", self.export)):
                try:
                    r.bind_all("<%s-%s>" % (mod, key), lambda e, fn=fn: fn())
                except Exception:
                    pass
        r.bind_all("<Escape>", lambda e: self._cancel_level())

    def _text_window(self, title, text):
        tk, ttk = self.tk, self.ttk
        w = tk.Toplevel(self.root)
        w.title(title)
        w.geometry("700x620")
        t = tk.Text(w, wrap="word", font="TkFixedFont", padx=12, pady=10)
        t.insert("1.0", text)
        t.configure(state="disabled")
        t.pack(fill="both", expand=True)
        ttk.Button(w, text="Close", command=w.destroy).pack(pady=6)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _float(var, default):
        try:
            return float(var.get())
        except Exception:
            return default

    @staticmethod
    def _int(var, default):
        try:
            return int(round(float(var.get())))
        except Exception:
            return default

    def _pal_key(self):
        return self.label_to_key.get(self.palette_var.get(), "mard221")

    def _pitch(self):
        return max(0.5, min(30.0, self._float(self.pitch_var, 2.6)))

    def _board(self):
        return max(4, min(400, self._int(self.board_var, 52)))

    def _rot(self):
        return max(-45.0, min(45.0, self._float(self.rot_var, 0.0)))

    def collect_settings(self):
        return Settings(
            palette=self._pal_key(), width=max(4, min(400, self._int(self.width_var, 52))),
            pitch_mm=self._pitch(), board=self._board(), max_colors=max(1, self._int(self.maxcol_var, 16)),
            method=self.method_var.get(), detail=self._int(self.detail_var, 2),
            rotation=self._rot(), flip=bool(self.flip_var.get()),
            crop=tuple(self.crop), offset_x=self._float(self.offx_var, 0.0), offset_y=self._float(self.offy_var, 0.0),
            brightness=self._float(self.bright_var, 1.0), contrast=self._float(self.contrast_var, 1.0),
            saturation=self._float(self.sat_var, 1.0), remove_bg=bool(self.bg_var.get()),
            bg_tolerance=self._float(self.bgtol_var, 12.0), cleanup=self._int(self.clean_var, 1),
            outline=bool(self.outline_var.get()), outline_code=self.outline_code_var.get(),
            only_codes=self.only_var.get(), exclude_codes=self.excl_var.get(),
            allow_special=bool(self.special_var.get()))

    # ------------------------------------------------------------------ events: settings
    def _on_change(self, *_):
        if self._suspend:
            return
        self._update_size_readout()
        self._draw_photo_overlay()
        self.schedule()

    def _on_rot_change(self, *_):
        if self._suspend:
            return
        new = self._rot()
        old = self._last_rot
        if abs(new - old) < 1e-9:
            return
        if self.src_preview is not None:
            w, h = self.src_preview.size
            W1, H1 = rotated_size(w, h, old)
            W2, H2 = rotated_size(w, h, new)
            x0, y0, x1, y1 = self.crop
            cx, cy = (x0 + x1) / 2.0 * W1, (y0 + y1) / 2.0 * H1
            cw, ch = (x1 - x0) * W1, (y1 - y0) * H1
            ncx, ncy = remap_point_rotation(cx, cy, w, h, old, new)
            cw, ch = min(cw, W2), min(ch, H2)
            nx0 = min(max(0.0, ncx - cw / 2.0), W2 - cw)
            ny0 = min(max(0.0, ncy - ch / 2.0), H2 - ch)
            self.crop = (nx0 / W2, ny0 / H2, (nx0 + cw) / W2, (ny0 + ch) / H2)
        self._last_rot = new
        self._update_display_geo()
        self.refresh_photo()
        self._on_change()

    def _on_flip_change(self, *_):
        if self._suspend:
            return
        flip = bool(self.flip_var.get())
        if flip != self._last_flip:
            x0, y0, x1, y1 = self.crop
            self.crop = (1.0 - x1, y0, 1.0 - x0, y1)
            self._last_flip = flip
        self._update_display_geo()
        self.refresh_photo()
        self._on_change()

    def _on_palette_change(self, init=False):
        pal = self.pals[self._pal_key()]
        for label, pitch, board in BEAD_TYPES:
            if abs(pitch - pal.mm) < 1e-6:
                if self.beadtype_var.get() != label:
                    self.beadtype_var.set(label)
                break
        values = []
        self._paint_codes = []
        for i in range(len(pal)):
            values.append(pal.describe(i))
            self._paint_codes.append(i)
        self.paint_combo.configure(values=values)
        self.cur_color = None
        self._update_swatch()
        if not init:
            self._on_change()

    def _on_beadtype(self, *_):
        for label, pitch, board in BEAD_TYPES:
            if label == self.beadtype_var.get():
                self._suspend = True
                self.pitch_var.set(str(pitch))
                if board:
                    self.board_var.set(str(board))
                self._suspend = False
                self._on_change()
                break

    def _on_aspect(self, *_):
        a = ASPECTS.get(self.aspect_var.get())
        if a is None or self.disp_size is None:
            return
        iw, ih = self.disp_size
        x0, y0, x1, y1 = self.crop
        w, h = (x1 - x0) * iw, (y1 - y0) * ih
        if w / max(h, 1e-6) > a:
            w = h * a
        else:
            h = w / a
        cx, cy = (x0 + x1) / 2.0 * iw, (y0 + y1) / 2.0 * ih
        self.crop = ((cx - w / 2) / iw, (cy - h / 2) / ih, (cx + w / 2) / iw, (cy + h / 2) / ih)
        self._draw_photo_overlay()
        self._on_change()

    def _nudge_rot(self, d):
        self.rot_var.set(round(self._rot() + d, 2))

    def _reset_adjust(self):
        self._suspend = True
        for v in (self.bright_var, self.contrast_var, self.sat_var):
            v.set(1.0)
        self._suspend = False
        self._on_change()

    def _set_target_cm(self):
        try:
            cm = float(self.target_cm_var.get().replace(",", "."))
        except ValueError:
            self.status_var.set("Type a width in centimetres, for example 15")
            return
        beads = int(round(cm * 10.0 / self._pitch()))
        self.width_var.set(max(8, min(240, beads)))

    def _estimated_dims(self):
        W = max(4, self._int(self.width_var, 52))
        if self.disp_size is None:
            return W, W
        iw, ih = self.disp_size
        _, H, _, _, _ = grid_geometry(iw, ih, self.crop, W)
        return W, H

    def _update_size_readout(self):
        W, H = self._estimated_dims()
        if self.outline_var.get():
            W, H = W + 2, H + 2
        pitch, board = self._pitch(), self._board()
        wcm, hcm = W * pitch / 10.0, H * pitch / 10.0
        bw, bh = math.ceil(W / board), math.ceil(H / board)
        if W < 30:
            detail = "icon / keychain level detail"
        elif W < 45:
            detail = "simple face, big features only"
        elif W < 70:
            detail = "faces read well"
        elif W < 110:
            detail = "detailed portrait"
        else:
            detail = "photo-like, a big project"
        self.size_var.set("%d x %d beads = %s pegs\nabout %.1f x %.1f cm  (%.1f x %.1f in)\n"
                          "boards: %d x %d of %dx%d\ndetail: %s" % (
                              W, H, format(W * H, ","), wcm, hcm, wcm / 2.54, hcm / 2.54, bw, bh, board, board,
                              detail))

    # ------------------------------------------------------------------ processing
    def schedule(self, delay=220):
        if self.src is None:
            return
        if self._job is not None:
            try:
                self.root.after_cancel(self._job)
            except Exception:
                pass
        self._job = self.root.after(delay, self._start_processing)

    def _start_processing(self):
        self._job = None
        if self.src is None:
            return
        if self._busy:
            self._pending = True
            return
        s = self.collect_settings()
        self._busy = True
        self._token += 1
        self.status_var.set("Working...")
        threading.Thread(target=self._worker, args=(self._token, s, self.src), daemon=True).start()

    def _worker(self, token, s, src):
        try:
            key = (id(src), round(s.rotation, 3), s.flip, round(s.brightness, 3), round(s.contrast, 3),
                   round(s.saturation, 3))
            with self._geo_lock:
                cached = self._geo_cache if self._geo_cache and self._geo_cache[0] == key else None
            if cached:
                img = cached[1]
            else:
                img = adjust_colors(transform_source(src, s.rotation, s.flip), s.brightness, s.contrast, s.saturation)
                with self._geo_lock:
                    self._geo_cache = (key, img)
            pat = build_pattern(img, s, self.pals[s.palette])
            self._q.put((token, pat, s))
        except Exception as exc:  # report, keep the app alive
            import traceback
            traceback.print_exc()
            self._q.put((token, exc, None))

    def _poll(self):
        try:
            while True:
                token, res, s = self._q.get_nowait()
                self._busy = False
                if isinstance(res, Exception):
                    self.status_var.set("Something went wrong: %s" % res)
                elif token == self._token:
                    self._apply_result(res, s)
                if self._pending:
                    self._pending = False
                    self._start_processing()
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def _apply_result(self, pat, s):
        sig = (pat.grid.shape, s.palette, tuple(round(v, 5) for v in s.crop), round(s.rotation, 3), s.flip,
               round(s.offset_x, 3), round(s.offset_y, 3), s.outline, s.width)
        msg = ""
        if self.edits and sig != self.edit_sig:
            self.edits.clear()
            self.undo.clear()
            msg = "  (Your hand edits were cleared because the size, crop or brand changed.)"
        self.edit_sig = sig
        self.base_grid = pat.grid.copy()
        for (r, c), v in self.edits.items():
            if 0 <= r < pat.H and 0 <= c < pat.W:
                pat.grid[r, c] = v
        self.pattern = pat
        self.last_settings = s
        if self._last_shape != pat.grid.shape:
            self._last_shape = pat.grid.shape
            self.fit_zoom(render=False)
        self._after_grid_change()
        self._update_size_readout()
        n = len(pat.used_sorted())
        self.status_var.set("Pattern ready: %d x %d beads, %d colors (%.2fs).%s" % (pat.W, pat.H, n, pat.elapsed, msg))

    def _after_grid_change(self, legend=True):
        pat = self.pattern
        if pat is None:
            return
        sizes, weak = analyze_structure(pat.grid)
        self.weak = weak
        if legend:
            self._refresh_legend()
        lines = []
        if len(sizes) > 1:
            small = sizes[1:]
            lines.append("%d separate pieces (main piece %s beads, others: %s). Loose pieces fall off "
                         "after ironing unless you join them." % (
                             len(sizes), format(sizes[0], ","), ", ".join(str(x) for x in small[:8]) +
                             ("..." if len(small) > 8 else "")))
        if weak:
            lines.append("%d spot(s) where parts touch only at a corner (red circles). Add a bead beside "
                         "each so they fuse." % len(weak))
        ok = not lines
        if ok and sizes:
            lines.append("Structure OK: one solid piece, no corner-only joints.")
        lines.extend(pat.notes)
        self.checks_lbl.configure(text="\n\n".join(lines), foreground="#166534" if ok else "#9a3412")
        self.refresh_pattern()

    # ------------------------------------------------------------------ legend
    def _refresh_legend(self):
        pat = self.pattern
        lb = self.legend
        lb.delete(0, "end")
        self._legend_idx = []
        labels = pattern_labels(pat)
        for idx, cnt in pat.used_sorted():
            code, name = pat.pal.codes[idx], pat.pal.names[idx]
            lab = labels.get(idx, code)
            if lab == code:
                text = " %-5s %-12s %6s" % (code, (name if name != code else "")[:12], format(cnt, ","))
            else:
                text = " %3s %-9s %-7s%5s" % (lab, code[:9], name[:7], format(cnt, ","))
            lb.insert("end", text)
            rgb = tuple(int(v) for v in pat.pal.rgb[idx])
            hexc = "#%02x%02x%02x" % rgb
            fg = "#%02x%02x%02x" % ink_for(rgb)
            lb.itemconfigure(lb.size() - 1, background=hexc, foreground=fg)
            self._legend_idx.append(idx)
        bw, bh = pat.boards()
        self.totals_lbl.configure(text="%s beads  |  %d colors\n%d x %d beads, boards %d x %d\n%s" % (
            format(pat.total(), ","), len(self._legend_idx), pat.W, pat.H, bw, bh, pat.size_text()))

    def _on_legend_select(self, _e=None):
        sel = self.legend.curselection()
        if not sel:
            return
        idx = self._legend_idx[sel[0]]
        self._set_cur_color(idx)
        if self.hilite_var.get():
            self.refresh_pattern()

    def _on_paint_combo(self, _e=None):
        i = self.paint_combo.current()
        if 0 <= i < len(self._paint_codes):
            self._set_cur_color(self._paint_codes[i])

    def _set_cur_color(self, idx):
        self.cur_color = idx
        self._update_swatch()

    def _update_swatch(self):
        pal = self.pals[self._pal_key()]
        if self.cur_color is None or self.cur_color >= len(pal):
            self.swatch.configure(text="  none  ", bg="#ffffff", fg="#000000")
            return
        rgb = tuple(int(v) for v in pal.rgb[self.cur_color])
        self.swatch.configure(text=pal.codes[self.cur_color], bg="#%02x%02x%02x" % rgb,
                              fg="#%02x%02x%02x" % ink_for(rgb))
        self.paint_combo_var.set(pal.describe(self.cur_color))

    # ------------------------------------------------------------------ photo tab
    def open_photo(self, path=None):
        if path is None:
            exts = "*.jpg *.jpeg *.png *.webp *.bmp *.gif *.tif *.tiff"
            if HEIF_OK:
                exts += " *.heic *.heif"
            path = self.fd.askopenfilename(title="Open photo", filetypes=[("Images", exts), ("All files", "*.*")])
            if not path:
                return
        try:
            img = load_photo(path)
        except Exception as exc:
            msg = str(exc)
            if path.lower().endswith((".heic", ".heif")) and not HEIF_OK:
                msg = "To open HEIC photos run:  pip install pillow-heif\n(or export the photo as JPG first)"
            self.mb.showerror("Could not open photo", msg)
            return
        self.src = img
        self.src_path = path
        prev = img.copy()
        prev.thumbnail((1100, 1100))
        self.src_preview = prev
        self._suspend = True
        self._last_rot = 0.0
        self._last_flip = False
        self.rot_var.set(0.0)
        self.flip_var.set(False)
        self.offx_var.set(0.0)
        self.offy_var.set(0.0)
        self._suspend = False
        self.edits.clear()
        self.undo.clear()
        self.pattern = None
        self._last_shape = None
        with self._geo_lock:
            self._geo_cache = None
        self.disp_key = None
        self._update_display_geo()
        self.crop = (0.0, 0.0, 1.0, 1.0)
        if ASPECTS.get(self.aspect_var.get()):
            self._on_aspect()
        self.file_lbl.configure(text=os.path.basename(path))
        self.nb.select(self.photo_tab)
        self.refresh_photo()
        self._on_change()
        self.status_var.set("Loaded. Drag on the photo to crop. Face tilted? Click Level eyes.")

    def _update_display_geo(self):
        if self.src_preview is None:
            self.disp_geo = None
            self.disp_size = None
            return
        key = (id(self.src_preview), round(self._rot(), 3), bool(self.flip_var.get()))
        if key == self.disp_key:
            return
        g = transform_source(self.src_preview, self._rot(), bool(self.flip_var.get()))
        bg = Image.new("RGBA", g.size, (70, 70, 74, 255))
        bg.alpha_composite(g)
        self.disp_geo = bg.convert("RGB")
        self.disp_size = g.size
        self.disp_key = key

    def refresh_photo(self, *_):
        c = self.photo_canvas
        cw, ch = max(60, c.winfo_width()), max(60, c.winfo_height())
        c.delete("all")
        if self.disp_geo is None:
            c.create_text(cw / 2, ch / 2 - 10, text="Open a photo to start   (Ctrl/Cmd + O)", fill="#e0e0e0",
                          font=self.font_sub)
            c.create_text(cw / 2, ch / 2 + 18, text="Everything runs on this computer. Nothing is uploaded.",
                          fill="#a0a0a0")
            self._photo_xf = None
            return
        iw, ih = self.disp_geo.size
        s = min((cw - 24) / float(iw), (ch - 24) / float(ih))
        dw, dh = max(1, int(iw * s)), max(1, int(ih * s))
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        key = (self.disp_key, dw, dh)
        if key != self._photo_base_key:
            self._photo_tk = self.ImageTk.PhotoImage(self.disp_geo.resize((dw, dh), LANCZOS))
            self._photo_base_key = key
        c.create_image(ox, oy, image=self._photo_tk, anchor="nw")
        self._photo_xf = (ox, oy, dw, dh)
        self._draw_photo_overlay()

    def _draw_photo_overlay(self):
        c = self.photo_canvas
        c.delete("ov")
        if self._photo_xf is None or self.disp_geo is None:
            return
        ox, oy, dw, dh = self._photo_xf
        x0, y0, x1, y1 = self.crop
        X0, Y0, X1, Y1 = ox + x0 * dw, oy + y0 * dh, ox + x1 * dw, oy + y1 * dh
        if self.showgrid_var.get():
            W = max(4, self._int(self.width_var, 52))
            _, H, cs, gx0, gy0 = grid_geometry(dw, dh, self.crop, W, self._float(self.offx_var, 0.0),
                                               self._float(self.offy_var, 0.0))
            gx0 += ox
            gy0 += oy
            step = 1 if cs >= 4 else (5 if cs * 5 >= 6 else 0)
            if step:
                for i in range(0, W + 1, step):
                    x = gx0 + i * cs
                    c.create_line(x, gy0, x, gy0 + H * cs, fill="#ffffff" if i % 5 == 0 else "#a7a7a7",
                                  tags="ov")
                for j in range(0, H + 1, step):
                    y = gy0 + j * cs
                    c.create_line(gx0, y, gx0 + W * cs, y, fill="#ffffff" if j % 5 == 0 else "#a7a7a7", tags="ov")
            mid = gx0 + W * cs / 2.0
            c.create_line(mid, gy0, mid, gy0 + H * cs, fill="#ff4d4d", dash=(6, 4), width=1, tags="ov")
        c.create_rectangle(X0, Y0, X1, Y1, outline="#000000", width=3, tags="ov")
        c.create_rectangle(X0, Y0, X1, Y1, outline="#22d3ee", width=1, tags="ov")
        for (px, py) in self.level_pts:
            c.create_oval(px - 6, py - 6, px + 6, py + 6, outline="#facc15", width=2, tags="ov")
        if len(self.level_pts) == 1 and self.level_mode:
            px, py = self.level_pts[0]
            c.create_text(px, py - 16, text="now the other eye", fill="#facc15", tags="ov")

    def _photo_norm(self, x, y):
        ox, oy, dw, dh = self._photo_xf
        return min(1.0, max(0.0, (x - ox) / float(dw))), min(1.0, max(0.0, (y - oy) / float(dh)))

    def _on_photo_press(self, e):
        if self._photo_xf is None:
            return
        if self.level_mode:
            self._level_click(e)
            return
        nx, ny = self._photo_norm(e.x, e.y)
        x0, y0, x1, y1 = self.crop
        full = (x1 - x0) > 0.995 and (y1 - y0) > 0.995
        if x0 < nx < x1 and y0 < ny < y1 and not full:
            self._drag = ("move", nx, ny, self.crop)
        else:
            self._drag = ("new", nx, ny, self.crop)

    def _on_photo_motion(self, e):
        if not self._drag or self._photo_xf is None:
            return
        nx, ny = self._photo_norm(e.x, e.y)
        kind, sx, sy, start = self._drag
        if kind == "move":
            x0, y0, x1, y1 = start
            w, h = x1 - x0, y1 - y0
            nx0 = min(max(0.0, x0 + nx - sx), 1.0 - w)
            ny0 = min(max(0.0, y0 + ny - sy), 1.0 - h)
            self.crop = (nx0, ny0, nx0 + w, ny0 + h)
        else:
            self.crop = self._make_rect(sx, sy, nx, ny)
        self._draw_photo_overlay()
        self._update_size_readout()

    def _on_photo_release(self, e):
        if not self._drag:
            return
        kind, sx, sy, start = self._drag
        self._drag = None
        ox, oy, dw, dh = self._photo_xf
        x0, y0, x1, y1 = self.crop
        if (x1 - x0) * dw < 10 or (y1 - y0) * dh < 10:
            self.crop = start
        self._draw_photo_overlay()
        self._on_change()

    def _make_rect(self, sx, sy, ex, ey):
        ox, oy, dw, dh = self._photo_xf
        a = ASPECTS.get(self.aspect_var.get())
        x0p, y0p, x1p, y1p = sx * dw, sy * dh, ex * dw, ey * dh
        if a:
            w, h = abs(x1p - x0p), abs(y1p - y0p)
            if w / max(h, 1e-6) > a:
                w = h * a
            else:
                h = w / a
            x1p = x0p + (w if x1p >= x0p else -w)
            y1p = y0p + (h if y1p >= y0p else -h)
        xa, xb = sorted((x0p, x1p))
        ya, yb = sorted((y0p, y1p))
        return (xa / dw, ya / dh, xb / dw, yb / dh)

    def reset_crop(self):
        self.crop = (0.0, 0.0, 1.0, 1.0)
        if ASPECTS.get(self.aspect_var.get()):
            self._on_aspect()
        else:
            self._draw_photo_overlay()
            self._on_change()

    def start_level(self):
        if self.disp_geo is None:
            self.status_var.set("Open a photo first.")
            return
        self.level_mode = True
        self.level_pts = []
        self.nb.select(self.photo_tab)
        self.photo_canvas.configure(cursor="crosshair")
        self.status_var.set("Level eyes: click the center of one eye, then the center of the other (Esc cancels).")
        self._draw_photo_overlay()

    def _cancel_level(self):
        if self.level_mode:
            self.level_mode = False
            self.level_pts = []
            self.photo_canvas.configure(cursor="tcross")
            self._draw_photo_overlay()
            self.status_var.set("Level eyes cancelled.")

    def _level_click(self, e):
        self.level_pts.append((e.x, e.y))
        self._draw_photo_overlay()
        if len(self.level_pts) < 2:
            return
        (x1, y1), (x2, y2) = sorted(self.level_pts)
        self.level_mode = False
        self.level_pts = []
        self.photo_canvas.configure(cursor="tcross")
        if abs(x2 - x1) < 4:
            self.status_var.set("Those points are too close together. Click Level eyes and try again.")
            self._draw_photo_overlay()
            return
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        new = max(-45.0, min(45.0, self._rot() + ang))
        self.rot_var.set(round(new, 2))
        self.status_var.set("Rotated %+.1f° so the eyes are level. Now nudge the grid so each eye sits in a bead."
                            % ang)

    # ------------------------------------------------------------------ pattern tab
    def fit_zoom(self, render=True):
        if self.pattern is None:
            return
        c = self.pat_canvas
        cw, ch = max(100, c.winfo_width()), max(100, c.winfo_height())
        cell = int(min((cw - 8) / float(self.pattern.W), (ch - 8) / float(self.pattern.H)))
        cell = max(2, min(40, cell, max_cell_for(self.pattern)))
        self._hold_render = not render
        try:
            self.zoom_var.set(cell)
        finally:
            self._hold_render = False

    def refresh_pattern(self, *_):
        if getattr(self, "_hold_render", False):
            return
        c = self.pat_canvas
        if self.pattern is None:
            c.delete("all")
            c.create_text(max(200, c.winfo_width() / 2), 120, fill="#888",
                          text="Your pattern appears here once a photo is open.")
            return
        cell = max(2, min(self._int(self.zoom_var, 10), max_cell_for(self.pattern)))
        style = self.style_var.get()
        hl = self.cur_color if self.hilite_var.get() else None
        labels = pattern_labels(self.pattern) if style == "codes" else None
        img = render_preview(self.pattern, cell, style, bool(self.boards_var.get()), hl, labels,
                             grid=bool(self.gridlines_var.get()))
        self._pattern_tk = self.ImageTk.PhotoImage(img)
        c.delete("all")
        c.create_image(0, 0, image=self._pattern_tk, anchor="nw")
        c.configure(scrollregion=(0, 0, img.width, img.height))
        self._render_cell = cell
        if self.weak_var.get():
            rad = max(4.0, cell * 0.55)
            for (r, cc) in self.weak[:1500]:
                x, y = cc * cell, r * cell
                c.create_oval(x - rad, y - rad, x + rad, y + rad, outline="#ff2d55", width=2)

    def _cell_at(self, e):
        if self.pattern is None:
            return None
        c = self.pat_canvas
        x, y = c.canvasx(e.x), c.canvasy(e.y)
        col, row = int(x // self._render_cell), int(y // self._render_cell)
        if 0 <= row < self.pattern.H and 0 <= col < self.pattern.W:
            return row, col
        return None

    def _on_pat_hover(self, e):
        rc = self._cell_at(e)
        if rc is None or self._stroke is not None:
            return
        r, c = rc
        v = int(self.pattern.grid[r, c])
        if v < 0:
            what = "no bead"
        else:
            cnt = int((self.pattern.grid == v).sum())
            what = "%s  (%s used)" % (self.pattern.pal.describe(v), format(cnt, ","))
        self.status_var.set("Column %d, row %d:  %s" % (c + 1, r + 1, what))

    def _on_pat_press(self, e, erase):
        rc = self._cell_at(e)
        if rc is None:
            return
        tool = "erase" if erase else self.tool_var.get()
        if tool == "pick":
            v = int(self.pattern.grid[rc])
            if v >= 0:
                self._set_cur_color(v)
                self.status_var.set("Picked %s" % self.pattern.pal.describe(v))
            return
        if tool in ("paint", "fill") and self.cur_color is None:
            self.status_var.set("Choose a color first: click one in the list on the right, or use Pick.")
            return
        self._stroke = []
        self._stroke_tool = tool
        if tool == "fill":
            self._fill(rc)
            self._end_stroke()
            return
        self._paint(rc, tool)
        self.refresh_pattern()

    def _on_pat_drag(self, e):
        if self._stroke is None or self._stroke_tool not in ("paint", "erase"):
            return
        rc = self._cell_at(e)
        if rc is not None and self._paint(rc, self._stroke_tool):
            self.refresh_pattern()

    def _on_pat_release(self, _e):
        if self._stroke is not None:
            self._end_stroke()

    def _end_stroke(self):
        if self._stroke:
            self.undo.append(self._stroke)
        self._stroke = None
        self._after_grid_change()

    def _targets(self, rc):
        r, c = rc
        out = [(r, c)]
        if self.mirror_var.get():
            mc = self.pattern.W - 1 - c
            if mc != c:
                out.append((r, mc))
        return out

    def _set_cell(self, r, c, v):
        g = self.pattern.grid
        if int(g[r, c]) == v:
            return False
        self._stroke.append(((r, c), self.edits.get((r, c), None), int(g[r, c])))
        self.edits[(r, c)] = v
        g[r, c] = v
        return True

    def _paint(self, rc, tool):
        v = -1 if tool == "erase" else int(self.cur_color)
        changed = False
        for r, c in self._targets(rc):
            changed |= self._set_cell(r, c, v)
        return changed

    def _fill(self, rc):
        g = self.pattern.grid
        H, W = g.shape
        v = int(self.cur_color)
        for r0, c0 in self._targets(rc):
            old = int(g[r0, c0])
            if old == v:
                continue
            stack, seen = [(r0, c0)], {(r0, c0)}
            while stack:
                r, c = stack.pop()
                self._set_cell(r, c, v)
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen and int(g[nr, nc]) == old:
                        seen.add((nr, nc))
                        stack.append((nr, nc))

    def undo_edit(self):
        if not self.undo or self.pattern is None:
            self.status_var.set("Nothing to undo.")
            return
        stroke = self.undo.pop()
        g = self.pattern.grid
        for (r, c), prev_edit, prev_val in reversed(stroke):
            if prev_edit is None:
                self.edits.pop((r, c), None)
            else:
                self.edits[(r, c)] = prev_edit
            if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                g[r, c] = prev_val
        self._after_grid_change()
        self.status_var.set("Undone.")

    def clear_edits(self):
        if self.pattern is None or self.base_grid is None:
            return
        self.edits.clear()
        self.undo.clear()
        self.pattern.grid = self.base_grid.copy()
        self._after_grid_change()
        self.status_var.set("All hand edits cleared.")

    # ------------------------------------------------------------------ wheel
    def _on_wheel(self, e):
        num = getattr(e, "num", None)
        if num == 4:
            d = -1
        elif num == 5:
            d = 1
        else:
            delta = getattr(e, "delta", 0)
            if not delta:
                return
            d = -1 if delta > 0 else 1
        w = self.root.winfo_containing(e.x_root, e.y_root)
        if w is None:
            return
        path = str(w)
        if path.startswith(str(self.lcanvas)):
            self.lcanvas.yview_scroll(d * 3, "units")
        elif path.startswith(str(self.pat_canvas)):
            if e.state & 0x0004 or e.state & 0x0008:   # Ctrl / Cmd zooms
                self.zoom_var.set(max(2, min(40, self._int(self.zoom_var, 10) - d)))
            elif e.state & 0x0001:
                self.pat_canvas.xview_scroll(d * 3, "units")
            else:
                self.pat_canvas.yview_scroll(d * 3, "units")

    # ------------------------------------------------------------------ export
    def export(self):
        if self.pattern is None:
            self.mb.showinfo(APP_NAME, "Open a photo first.")
            return
        stem = os.path.splitext(os.path.basename(self.src_path or "pattern"))[0]
        path = self.fd.asksaveasfilename(
            title="Export: choose a name (several files will be created)", initialfile="%s_pattern" % stem,
            defaultextension=".pdf", filetypes=[("Pattern files", "*.pdf"), ("All files", "*.*")])
        if not path:
            return
        base = os.path.splitext(path)[0]
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            files, warnings = export_all(self.pattern, base, self.paper_var.get(), self.mode_var.get(), stem)
        except Exception as exc:
            self.root.configure(cursor="")
            self.mb.showerror("Export failed", str(exc))
            return
        self.root.configure(cursor="")
        folder = os.path.dirname(os.path.abspath(base))
        msg = "Saved in %s:\n\n%s" % (folder, "\n".join("  " + os.path.basename(f) for f in files))
        if warnings:
            msg += "\n\n" + "\n".join(warnings)
        msg += ("\n\nTip: print *_true_size.pdf at 100%% (Actual size) and check the ruler."
                "\n\nOpen the folder now?")
        if self.mb.askyesno("Export done", msg):
            open_folder(folder)
        self.status_var.set("Exported %d files to %s" % (len(files), folder))


# =============================================================================
# Entry points
# =============================================================================
def run_cli(args):
    pals = palettes()
    if args.palette not in pals:
        print("Unknown palette. Choose one of: %s" % ", ".join(pals))
        return 2
    pal = pals[args.palette]
    pitch = args.pitch or pal.mm
    s = Settings(palette=args.palette, width=args.width, pitch_mm=pitch, board=args.board or default_board_for(pitch),
                 max_colors=args.colors, method=args.method, detail=args.detail, rotation=args.rotate, flip=args.flip,
                 offset_x=args.nudge_x, offset_y=args.nudge_y, remove_bg=args.remove_bg, bg_tolerance=args.bg_tol,
                 cleanup=args.cleanup, outline=args.outline, only_codes=args.only or "",
                 exclude_codes=args.exclude or "", brightness=args.brightness, contrast=args.contrast,
                 saturation=args.saturation)
    img = load_photo(args.photo)
    img = adjust_colors(transform_source(img, s.rotation, s.flip), s.brightness, s.contrast, s.saturation)
    pat = build_pattern(img, s, pal)
    stem = os.path.splitext(os.path.basename(args.photo))[0]
    mode = {"board": CHART_MODES[0], "whole": CHART_MODES[1], "large": CHART_MODES[2]}[args.chart]
    files, warnings = export_all(pat, args.export, args.paper, mode, stem)
    sizes, weak = analyze_structure(pat.grid)
    print("%d x %d beads, %d colors, %s beads, %s (%.2fs)" % (
        pat.W, pat.H, len(pat.used_sorted()), format(pat.total(), ","), pat.size_text(), pat.elapsed))
    print("pieces: %d, corner-only joints: %d" % (len(sizes), len(weak)))
    for n in pat.notes + warnings:
        print("note:", n)
    for f in files:
        print("wrote", f)
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="%s: photo to 拼豆 bead pattern, fully offline." % APP_NAME)
    ap.add_argument("photo", nargs="?", help="photo to open")
    ap.add_argument("--export", metavar="BASE", help="no window: write BASE_chart.pdf, BASE_true_size.pdf, ...")
    ap.add_argument("--palette", default="mard221", help="palette key, e.g. mard221 coco291 perler hama")
    ap.add_argument("--width", type=int, default=52, help="pattern width in beads")
    ap.add_argument("--colors", type=int, default=16, help="maximum number of colors")
    ap.add_argument("--method", choices=["dominant", "average"], default="dominant")
    ap.add_argument("--detail", type=int, default=2, choices=[0, 1, 2, 3], help="keep thin dark features")
    ap.add_argument("--pitch", type=float, default=None, help="bead/peg pitch in mm (default from palette)")
    ap.add_argument("--board", type=int, default=None, help="pegs per board side (default 52 mini, 29 midi)")
    ap.add_argument("--rotate", type=float, default=0.0)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--nudge-x", type=float, default=0.0)
    ap.add_argument("--nudge-y", type=float, default=0.0)
    ap.add_argument("--remove-bg", action="store_true")
    ap.add_argument("--bg-tol", type=float, default=12.0)
    ap.add_argument("--cleanup", type=int, default=1, choices=[0, 1, 2, 3])
    ap.add_argument("--outline", action="store_true")
    ap.add_argument("--only", help="only use these codes")
    ap.add_argument("--exclude", help="never use these codes")
    ap.add_argument("--brightness", type=float, default=1.0)
    ap.add_argument("--contrast", type=float, default=1.0)
    ap.add_argument("--saturation", type=float, default=1.0)
    ap.add_argument("--paper", choices=list(PAPERS), default="Letter")
    ap.add_argument("--chart", choices=["board", "whole", "large"], default="board")
    args = ap.parse_args(argv)
    if args.export:
        if not args.photo:
            ap.error("--export needs a photo")
        return run_cli(args)

    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    try:
        import tkinter as tk
    except ImportError:
        print("tkinter is missing. On macOS with Homebrew: brew install python-tk. "
              "On Linux: sudo apt install python3-tk. Or use the python.org installer.")
        return 1
    root = tk.Tk()
    PindouApp(root, args.photo)
    root.mainloop()
    return 0


# Bundled bead palettes (generated from open data, see the header). Keep this last.
PALETTE_JSON = r'''{"mard221":{"label":"MARD 221 (most common in China)","name":"MARD 221","mm":2.6,"source":"HansBug/pindou-color-data","colors":[["A1","","FAF5CD",0],["A2","","FCFED6",0],["A3","","FCFF92",0],["A4","","F7EC5C",0],["A5","","F0D83A",0],["A6","","FDA951",0],["A7","","FA8C4F",0],["A8","","FBDA4D",0],["A9","","F79D5F",0],["A10","","F47E38",0],["A11","","FEDB99",0],["A12","","FDA276",0],["A13","","FEC667",0],["A14","","F75842",0],["A15","","FBF65E",0],["A16","","FEFF97",0],["A17","","FDE173",0],["A18","","FCBF80",0],["A19","","FD7E77",0],["A20","","F9D666",0],["A21","","FAE393",0],["A22","","EDF878",0],["A23","","E4C8BA",0],["A24","","F3F6A9",0],["A25","","FDF785",0],["A26","","FFC734",0],["B1","","DFF13B",0],["B2","","64F343",0],["B3","","A1F586",0],["B4","","5FDF34",0],["B5","","39E158",0],["B6","","64E0A4",0],["B7","","3EAE7C",0],["B8","","1D9B54",0],["B9","","2A5037",0],["B10","","9AD1BA",0],["B11","","627032",0],["B12","","1A6E3D",0],["B13","","C8E87D",0],["B14","","ABE84F",0],["B15","","305335",0],["B16","","C0ED9C",0],["B17","","9EB33E",0],["B18","","E6ED4F",0],["B19","","26B78E",0],["B20","","CBECCF",0],["B21","","18616A",0],["B22","","0A4241",0],["B23","","343B1A",0],["B24","","E8FAA6",0],["B25","","4E846D",0],["B26","","907C35",0],["B27","","D0E0AF",0],["B28","","9EE5BB",0],["B29","","C6DF5F",0],["B30","","E3FBB1",0],["B31","","B4E691",0],["B32","","92AD60",0],["C1","","F0FEE4",0],["C2","","ABF8FE",0],["C3","","A2E0F7",0],["C4","","44CDFB",0],["C5","","06AADF",0],["C6","","54A7E9",0],["C7","","3977CA",0],["C8","","0F52BD",0],["C9","","3349C3",0],["C10","","3CBCE3",0],["C11","","2ADED3",0],["C12","","1E334E",0],["C13","","CDE7FE",0],["C14","","D5FCF7",0],["C15","","21C5C4",0],["C16","","1858A2",0],["C17","","02D1F3",0],["C18","","213244",0],["C19","","18869D",0],["C20","","1A70A9",0],["C21","","BCDDFC",0],["C22","","6BB1BB",0],["C23","","C8E2FD",0],["C24","","7EC5F9",0],["C25","","A9E8E0",0],["C26","","42ADCF",0],["C27","","D0DEF9",0],["C28","","BDCEE8",0],["C29","","364A89",0],["D1","","ACB7EF",0],["D2","","868DD3",0],["D3","","3554AF",0],["D4","","162D7B",0],["D5","","B34EC6",0],["D6","","B37BDC",0],["D7","","8758A9",0],["D8","","E3D2FE",0],["D9","","D5B9F4",0],["D10","","301A49",0],["D11","","BEB9E2",0],["D12","","DC99CE",0],["D13","","B5038D",0],["D14","","862993",0],["D15","","2F1F8C",0],["D16","","E2E4F0",0],["D17","","C7D3F9",0],["D18","","9A64B8",0],["D19","","D8C2D9",0],["D20","","9A35AD",0],["D21","","940595",0],["D22","","38389A",0],["D23","","EADBF8",0],["D24","","768AE1",0],["D25","","4950C2",0],["D26","","D6C6EB",0],["E1","","F6D4CB",0],["E2","","FCC1DD",0],["E3","","F6BDE8",0],["E4","","E8649E",0],["E5","","F0569F",0],["E6","","EB4172",0],["E7","","C53674",0],["E8","","FDDBE9",0],["E9","","E376C7",0],["E10","","D13B95",0],["E11","","F7DAD4",0],["E12","","F693BF",0],["E13","","B5026A",0],["E14","","FAD4BF",0],["E15","","F5C9CA",0],["E16","","FBF4EC",0],["E17","","F7E3EC",0],["E18","","F9C8DB",0],["E19","","F6BBD1",0],["E20","","D7C6CE",0],["E21","","C09DA4",0],["E22","","B38C9F",0],["E23","","937D8A",0],["E24","","DEBEE5",0],["F1","","FE9381",0],["F2","","F63D4B",0],["F3","","EE4E3E",0],["F4","","FB2A40",0],["F5","","E10328",0],["F6","","913635",0],["F7","","911932",0],["F8","","BB0126",0],["F9","","E0677A",0],["F10","","874628",0],["F11","","592323",0],["F12","","F3536B",0],["F13","","F45C45",0],["F14","","FCADB2",0],["F15","","D50527",0],["F16","","F8C0A9",0],["F17","","E89B7D",0],["F18","","D07F4A",0],["F19","","BE454A",0],["F20","","C69495",0],["F21","","F2B8C6",0],["F22","","F7C3D0",0],["F23","","ED806C",0],["F24","","E09DAF",0],["F25","","E84854",0],["G1","","FFE4D3",0],["G2","","FCC6AC",0],["G3","","F1C4A5",0],["G4","","DCB387",0],["G5","","E7B34E",0],["G6","","E3A014",0],["G7","","985C3A",0],["G8","","713D2F",0],["G9","","E4B685",0],["G10","","DA8C42",0],["G11","","DAC898",0],["G12","","FEC993",0],["G13","","B2714B",0],["G14","","8B684C",0],["G15","","F6F8E3",0],["G16","","F2D8C1",0],["G17","","77544E",0],["G18","","FFE3D5",0],["G19","","DD7D41",0],["G20","","A5452F",0],["G21","","B38561",0],["H1","","FFFFFF",0],["H2","","FBFBFB",0],["H3","","B4B4B4",0],["H4","","878787",0],["H5","","464648",0],["H6","","2C2C2C",0],["H7","","010101",0],["H8","","E7D6DC",0],["H9","","EFEDEE",0],["H10","","EBEBEB",0],["H11","","CDCDCD",0],["H12","","FDF6EE",0],["H13","","F4EDF1",0],["H14","","CED7D4",0],["H15","","9AA6A6",0],["H16","","1B1213",0],["H17","","F0EEEF",0],["H18","","FCFFF6",0],["H19","","F2EEE5",0],["H20","","96A09F",0],["H21","","F8FBE6",0],["H22","","CACAD2",0],["H23","","9B9C94",0],["M1","","BBC6B6",0],["M2","","909994",0],["M3","","697E81",0],["M4","","E0D4BC",0],["M5","","D1CCAF",0],["M6","","B0AA86",0],["M7","","B0A796",0],["M8","","AE8082",0],["M9","","A68862",0],["M10","","C4B3BB",0],["M11","","9D7693",0],["M12","","644B51",0],["M13","","C79266",0],["M14","","C27563",0],["M15","","747D7A",0]]},"mard291":{"label":"MARD 291 (221 + extended series)","name":"MARD 291","mm":2.6,"source":"HansBug/pindou-color-data","colors":[["A1","","FAF5CD",0],["A2","","FCFED6",0],["A3","","FCFF92",0],["A4","","F7EC5C",0],["A5","","FFE44B",0],["A6","","FDA951",0],["A7","","FA8C4F",0],["A8","","F9E045",0],["A9","","F99C5F",0],["A10","","F47E36",0],["A11","","FEDB99",0],["A12","","FDA276",0],["A13","","FEC667",0],["A14","","F85842",0],["A15","","FBF65E",0],["A16","","FEFF97",0],["A17","","FDE173",0],["A18","","FCBF80",0],["A19","","FD7E77",0],["A20","","F9D66E",0],["A21","","FAE393",0],["A22","","EDF878",0],["A23","","E1C9BD",0],["A24","","F3F6A9",0],["A25","","FFD785",0],["A26","","FEC832",0],["B1","","DFF139",0],["B2","","64F343",0],["B3","","9FF685",0],["B4","","5FDF34",0],["B5","","39E158",0],["B6","","64E0A4",0],["B7","","3FAE7C",0],["B8","","1D9E54",0],["B9","","2A5037",0],["B10","","9AD1BA",0],["B11","","627032",0],["B12","","1A6E3D",0],["B13","","C8E87D",0],["B14","","ACE84C",0],["B15","","305335",0],["B16","","C0ED9C",0],["B17","","9EB33E",0],["B18","","E6ED4F",0],["B19","","26B78E",0],["B20","","CAEDCF",0],["B21","","176268",0],["B22","","0A4241",0],["B23","","343B1A",0],["B24","","E8FAA6",0],["B25","","4E846D",0],["B26","","907C35",0],["B27","","D0E0AF",0],["B28","","9EE5BB",0],["B29","","C6DF5F",0],["B30","","E3FBB1",0],["B31","","B2E694",0],["B32","","92AD60",0],["C1","","E8FFE7",0],["C2","","ABF8FE",0],["C3","","9EE0F8",0],["C4","","44CDFB",0],["C5","","06ABE3",0],["C6","","54A7E9",0],["C7","","3977CC",0],["C8","","0F52BD",0],["C9","","3349C3",0],["C10","","3DBBE3",0],["C11","","2ADED3",0],["C12","","1E334E",0],["C13","","CDE7FE",0],["C14","","D6FDFC",0],["C15","","21C5C4",0],["C16","","1858A2",0],["C17","","02D1F3",0],["C18","","213244",0],["C19","","188690",0],["C20","","1A70A9",0],["C21","","BEDDFC",0],["C22","","6BB1BB",0],["C23","","C8E2F9",0],["C24","","7EC5F9",0],["C25","","A9E8E0",0],["C26","","42ADD1",0],["C27","","D0DEEF",0],["C28","","BDCEED",0],["C29","","364A89",0],["D1","","ACB7EF",0],["D2","","868DD3",0],["D3","","3653AF",0],["D4","","162C7E",0],["D5","","B34EC6",0],["D6","","B37BDC",0],["D7","","8758A9",0],["D8","","E3D2FE",0],["D9","","D6BAF5",0],["D10","","301A49",0],["D11","","BCBAE2",0],["D12","","DC99CE",0],["D13","","B5038F",0],["D14","","882893",0],["D15","","2F1E8E",0],["D16","","E2E4F0",0],["D17","","C7D3F9",0],["D18","","9A64B8",0],["D19","","D8C2D9",0],["D20","","9C34AD",0],["D21","","940595",0],["D22","","383995",0],["D23","","FADBF8",0],["D24","","768AE1",0],["D25","","4950C2",0],["D26","","D6C6EB",0],["E1","","F6D4CB",0],["E2","","FCC1DD",0],["E3","","F6BDE8",0],["E4","","E9639E",0],["E5","","F1559F",0],["E6","","EC4072",0],["E7","","C63674",0],["E8","","FDDBE9",0],["E9","","E575C7",0],["E10","","D33997",0],["E11","","F7DAD4",0],["E12","","F893BF",0],["E13","","B5026A",0],["E14","","FAD4BF",0],["E15","","F5C9CA",0],["E16","","FBF4EC",0],["E17","","F7E3EC",0],["E18","","FBCBDB",0],["E19","","F6BBD1",0],["E20","","D7C6CE",0],["E21","","C09DA4",0],["E22","","B58B9F",0],["E23","","937D8A",0],["E24","","DEBEE5",0],["F1","","FF9280",0],["F2","","F73D48",0],["F3","","EF4D3E",0],["F4","","F92B40",0],["F5","","E30328",0],["F6","","913635",0],["F7","","911932",0],["F8","","BB0126",0],["F9","","E0677A",0],["F10","","874628",0],["F11","","6F321D",0],["F12","","F8516D",0],["F13","","F45C45",0],["F14","","FCADB2",0],["F15","","D50527",0],["F16","","F8C0A9",0],["F17","","E89B7D",0],["F18","","D07E4A",0],["F19","","BE454A",0],["F20","","C69495",0],["F21","","F2BBC6",0],["F22","","F7C3D0",0],["F23","","EC806D",0],["F24","","E09DAF",0],["F25","","E84854",0],["G1","","FFE4D3",0],["G2","","FCC6AC",0],["G3","","F1C4A5",0],["G4","","DCB387",0],["G5","","E7B34E",0],["G6","","F3A014",0],["G7","","98503A",0],["G8","","4B2B1C",0],["G9","","E4B685",0],["G10","","DA8C42",0],["G11","","DAC898",0],["G12","","FEC993",0],["G13","","B2714B",0],["G14","","8B684C",0],["G15","","FCF9E0",0],["G16","","F2D8C1",0],["G17","","79544E",0],["G18","","FFE4D6",0],["G19","","DD7D41",0],["G20","","A5452F",0],["G21","","B38561",0],["H1","","FBFBFB",0],["H2","","FFFFFF",0],["H3","","B4B4B4",0],["H4","","878787",0],["H5","","464648",0],["H6","","2C2C2C",0],["H7","","010101",0],["H8","","E7D6DC",0],["H9","","EFEDEE",0],["H10","","ECEAEB",0],["H11","","CDCDCD",0],["H12","","FDF6EE",0],["H13","","F4EFD1",0],["H14","","CED7D4",0],["H15","","98A6A6",0],["H16","","1B1213",0],["H17","","F0EEEF",0],["H18","","FCFFF8",0],["H19","","F2EEE5",0],["H20","","96A09F",0],["H21","","FFFBE1",0],["H22","","CACADA",0],["H23","","9B9C94",0],["M1","","BBC6B6",0],["M2","","909994",0],["M3","","697E80",0],["M4","","E0D4BC",0],["M5","","D0CBAE",0],["M6","","B0AA86",0],["M7","","B0A796",0],["M8","","AE8082",0],["M9","","A88764",0],["M10","","C6B2BB",0],["M11","","9D7693",0],["M12","","644B51",0],["M13","","C79266",0],["M14","","C37463",0],["M15","","747D7A",0],["P1","","F9F9F9",1],["P2","","ABABAB",1],["P3","","B6DBAF",1],["P4","","FEA2A3",1],["P5","","EB903F",1],["P6","","63CEA2",1],["P7","","E79273",1],["P8","","ECDB59",1],["P9","","DBD9DA",1],["P10","","DBC7EA",1],["P11","","F1E9D4",1],["P12","","E9EDEE",1],["P13","","ADCBF1",1],["P14","","337BAD",1],["P15","","668575",1],["P16","","FDC24E",1],["P17","","FDA42E",1],["P18","","FEBDA7",1],["P19","","FFDEE9",1],["P20","","FCBFD1",1],["P21","","E8BEC2",1],["P22","","DFAAA4",1],["P23","","A3656A",1],["R1","","D40E1F",1],["R2","","F13484",1],["R3","","FB852B",1],["R4","","F8ED33",1],["R5","","32C958",1],["R6","","1EBA93",1],["R7","","1D779C",1],["R8","","1960C8",1],["R9","","945AB1",1],["R10","","F8DA54",1],["R11","","FCECF7",1],["R12","","D8D4D3",1],["R13","","56534E",1],["R14","","A3E7DC",1],["R15","","78CEE7",1],["R16","","3FCDCE",1],["R17","","4E8379",1],["R18","","7DCA9C",1],["R19","","C8E664",1],["R20","","E3CCBA",1],["R21","","A17140",1],["R22","","6B372C",1],["R23","","F6BB6F",1],["R24","","F3C6C0",1],["R25","","C76A62",1],["R26","","D093BC",1],["R27","","E58EAE",1],["R28","","9F85CF",1],["T1","","FCFDFF",1],["Y1","","FF6FB7",1],["Y2","","FDB583",1],["Y3","","D8FCA4",1],["Y4","","91DAFB",1],["Y5","","E987EA",1],["ZG1","","DAABB3",1],["ZG2","","D6AA87",1],["ZG3","","C1BD8D",1],["ZG4","","96B69F",1],["ZG5","","849DC6",1],["ZG6","","94BFE2",1],["ZG7","","E2A9D2",1],["ZG8","","AB91C0",1],["Q1","","F2A5E8",1],["Q2","","E9EC91",1],["Q3","","FFFF00",1],["Q4","","FFEBFA",1],["Q5","","76CEDE",1]]},"coco291":{"label":"COCO 可可 291","name":"COCO 291","mm":2.6,"source":"HansBug/pindou-color-data","colors":[["A01","","FFFFFF",0],["A02","","FEFCFF",0],["A03","","FFFDF7",0],["A04","","FFFCEE",0],["A05","","FFFBEE",0],["A06","","F1EADC",0],["A07","","FCFCF9",0],["A08","","EFF0EB",0],["A09","","FFF7FF",0],["A10","","F4F9FF",0],["A11","","FFFDF7",0],["A12","","F3F5F1",0],["B01","","D3DDDE",0],["B02","","D6E8EB",0],["B03","","AFBBBD",0],["B04","","A7BDC4",0],["B05","","7F8A8A",0],["B06","","5D6163",0],["B07","","3A3E40",0],["B08","","2B2B2B",0],["B09","","000000",0],["B10","","C3D1CE",0],["B11","","DCE0C8",0],["C01","","FF6C71",0],["C02","","FF6F60",0],["C03","","FF6464",0],["C04","","F66B4E",0],["C05","","FF8046",0],["C06","","F0393B",0],["C07","","E03638",0],["C08","","CC3133",0],["C09","","B82D2D",0],["C10","","783644",0],["D1","","FFDD94",0],["D2","","FFD3A1",0],["D3","","FFD64F",0],["D4","","FFC31F",0],["D5","","FFBA30",0],["D6","","FFAE82",0],["D7","","FF985C",0],["D8","","F28D51",0],["E01","","FFFFD5",0],["E02","","FFF7D2",0],["E03","","FFFFB3",0],["E04","","FFFA6A",0],["E05","","FFFF9B",0],["E06","","FFDA7C",0],["E07","","FFEB41",0],["E08","","FCCF3E",0],["E09","","FFC845",0],["E10","","FFDF91",0],["E11","","FFF3A2",0],["E12","","FFDFBA",0],["E13","","FFFFD0",0],["E14","","FFDD78",0],["E15","","FCBA49",0],["F01","","F1FFC2",0],["F02","","E3FC9A",0],["F03","","D2FCAE",0],["F04","","97F5A9",0],["F05","","E7F247",0],["F06","","C5E848",0],["F07","","F7FA46",0],["F08","","8ECF53",0],["F09","","6FEB67",0],["F10","","65C964",0],["F11","","3D8C57",0],["F12","","1D734E",0],["F13","","8FA335",0],["F14","","515C22",0],["F15","","2A4A2E",0],["F16","","27452B",0],["F17","","323815",0],["F18","","5B8265",0],["F19","","948037",0],["F20","","E0E8B5",0],["F21","","BCF7B1",0],["F22","","EDF256",0],["F23","","F6FCCB",0],["F24","","DEF7AF",0],["F25","","B5C26D",0],["G01","","E9FADF",0],["G02","","E9FFE3",0],["G03","","9EDBC8",0],["G04","","66E0B2",0],["G05","","189E7F",0],["G06","","56BD96",0],["G07","","3D8773",0],["G08","","28594C",0],["GB1","","FF4D5E",0],["GB2","","FF916E",0],["GB3","","F7F587",0],["GB4","","CBE38A",0],["GB5","","8E92E6",0],["GB6","","5193B5",0],["GB7","","D42D7B",0],["GB8","","7C76BD",0],["H01","","DEFAFC",0],["H02","","F0FEF2",0],["H03","","BDF6ED",0],["H04","","B7EEF7",0],["H05","","58CBF0",0],["H06","","E3FDFF",0],["H07","","00AACC",0],["H08","","64B2DB",0],["H09","","30CBE4",0],["H10","","00C5CF",0],["H11","","00BAB7",0],["H12","","2397A3",0],["H13","","198DB4",0],["H14","","0065A1",0],["H15","","1F5082",0],["H16","","0D55B3",0],["H17","","0787AC",0],["H18","","335F7D",0],["H19","","56D7E3",0],["H20","","304561",0],["H21","","49608F",0],["H22","","323578",0],["H23","","2C3B4A",0],["H24","","324752",0],["H25","","78C4BE",0],["H26","","DCF2EA",0],["H27","","92D4F0",0],["H28","","B2F0D7",0],["H29","","55B5C4",0],["H30","","DFF1FB",0],["H31","","BBE3F0",0],["H32","","4D6A86",0],["J01","","F2FAFF",0],["J02","","F5F2FF",0],["J03","","E7E6FF",0],["J04","","E1DCFC",0],["J05","","BCE3FB",0],["J06","","C3E1F7",0],["J07","","A8CCF0",0],["J08","","7DA1C8",0],["J09","","E5E5FA",0],["J10","","D7B1E3",0],["J11","","9F92CF",0],["J12","","B36BAA",0],["J13","","978BBF",0],["J14","","B15095",0],["J15","","56498F",0],["J16","","74458A",0],["J17","","9563B0",0],["J18","","6F4285",0],["J19","","07004A",0],["J20","","6C82C9",0],["J21","","3C46A1",0],["J22","","EAD5FA",0],["K01","","FFF8F3",0],["K02","","FFF1E9",0],["K03","","FADBD2",0],["K04","","FCEBE9",0],["K05","","FFE0C4",0],["K06","","FFE8D4",0],["K07","","FFBCB8",0],["K08","","FF9D8A",0],["K09","","FFC5A7",0],["K10","","FC9798",0],["K11","","FFF0F5",0],["K12","","FFEBDE",0],["K13","","FFE5E6",0],["K14","","FFE9F2",0],["K15","","FFE7F2",0],["K16","","FFC7DD",0],["K17","","F7A3C4",0],["K18","","F78FCE",0],["K19","","F26697",0],["K20","","E6758A",0],["K21","","E985B1",0],["K22","","FF4576",0],["K23","","C9427D",0],["K24","","AD5081",0],["K25","","AB3967",0],["K26","","EDE2F0",0],["K27","","E0C4C3",0],["K28","","D4A0B7",0],["K29","","A18A98",0],["K30","","FAF2FF",0],["K31","","F7B38C",0],["K32","","E08743",0],["K33","","BF5960",0],["K34","","D4A492",0],["K35","","FCC7CC",0],["K36","","FFD3BF",0],["K37","","FA9276",0],["K38","","FFCDD0",0],["K39","","E34447",0],["L1","","D40E1F",0],["L10","","F8DA54",0],["L11","","FCECF7",0],["L12","","D8D4D3",0],["L13","","56534E",0],["L14","","FFFFFF",0],["L2","","F13484",0],["L3","","FB852B",0],["L4","","F8ED33",0],["L5","","32C958",0],["L6","","1EBA93",0],["L7","","1D779C",0],["L8","","1960C8",0],["L9","","945AB1",0],["M01","","F9F9F9",0],["M02","","ABABAB",0],["M03","","B6DBAF",0],["M04","","FEA2A3",0],["M05","","EB903F",0],["M06","","63CEA2",0],["M07","","E79273",0],["M08","","ECDB59",0],["M09","","DBD9DA",0],["M10","","DBC7EA",0],["M11","","F1E9D4",0],["M12","","E9EDEE",0],["M13","","ADCBF1",0],["M14","","337BAD",0],["M15","","668575",0],["M16","","FDC24E",0],["M17","","FDA42E",0],["M18","","FEBDA7",0],["M19","","FFDEE9",0],["M20","","FCBFD1",0],["M21","","E8BEC2",0],["M22","","DFAAA4",0],["M23","","A3656A",0],["N1","","FF6FB7",0],["N2","","FDB583",0],["N3","","D8FCA4",0],["N4","","91DAFB",0],["N5","","E987EA",0],["S1","","A3E7DC",0],["S10","","F6BB6F",0],["S11","","F3C6C0",0],["S12","","C76A62",0],["S13","","D093BC",0],["S14","","E58EAE",0],["S15","","9F85CF",0],["S2","","78CEE7",0],["S3","","3FCDCE",0],["S4","","4E8379",0],["S5","","7DCA9C",0],["S6","","C8E664",0],["S7","","E3CCBA",0],["S8","","A17140",0],["S9","","6B372C",0],["W1","","FEFB00",0],["W2","","FFEBFA",0],["W3","","F2A5E8",0],["W4","","E9EC91",0],["W5","","76CEDE",0],["Y01","","D5E8D7",0],["Y02","","95B091",0],["Y03","","62858A",0],["Y04","","E0D6C3",0],["Y05","","DED7B1",0],["Y06","","BDB293",0],["Y07","","C9BCA9",0],["Y08","","B8A9A6",0],["Y09","","CCB88B",0],["Y10","","C2B5BC",0],["Y11","","AE99B0",0],["Y12","","5C5153",0],["Y13","","E0AC82",0],["Y14","","DE8D6D",0],["Y15","","95A19C",0],["Z01","","FFF3DE",0],["Z02","","FFEAC2",0],["Z03","","FDFCD9",0],["Z04","","FAE8C8",0],["Z05","","FFDDAB",0],["Z06","","F5C69B",0],["Z07","","F0E49E",0],["Z08","","E6C495",0],["Z09","","FCDA95",0],["Z10","","F0AC46",0],["Z11","","F5AE48",0],["Z12","","FAB646",0],["Z13","","F0D393",0],["Z14","","BD8A6A",0],["Z15","","BD9044",0],["Z16","","54483A",0],["Z17","","9E836C",0],["Z18","","A67E61",0],["Z19","","A36943",0],["Z20","","A16446",0],["Z21","","9E5B4E",0],["Z22","","57423B",0],["Z23","","5C3F3B",0],["Z24","","C7A58D",0]]},"manman278":{"label":"漫漫 Manman 278","name":"Manman 278","mm":2.6,"source":"HansBug/pindou-color-data","colors":[["A1","","FF9280",0],["A2","","F73D48",0],["A3","","EF4D3E",0],["A4","","F92B40",0],["A5","","F36328",0],["A6","","911932",0],["A7","","FCC1DD",0],["A10","","F1559F",0],["A20","","F893BF",0],["A11","","EC4072",0],["A12","","C63674",0],["A13","","FDDBE9",0],["A14","","E575C7",0],["A15","","E0677A",0],["A16","","D33997",0],["A17","","BB0126",0],["A18","","FDA276",0],["A19","","F7DAD4",0],["A21","","B5026A",0],["A22","","F8516D",0],["A23","","F5C9CA",0],["A24","","F45C45",0],["A25","","FCADB2",0],["B1","","FCFED6",0],["B2","","FCFF92",0],["B3","","F7EC5C",0],["B4","","F0D83A",0],["B5","","FDA951",0],["B6","","FA8C4F",0],["B7","","E7B34E",0],["B8","","F3A014",0],["B9","","DA8C42",0],["B10","","FDD94D",0],["B11","","F99C5F",0],["B12","","F47E36",0],["B13","","FEC667",0],["B14","","F85842",0],["B15","","FBF65E",0],["C1","","DFF139",0],["C2","","64F343",0],["C3","","5FDF34",0],["C4","","39E158",0],["C5","","1D9E54",0],["C6","","2A5037",0],["C7","","9FF685",0],["C8","","FFFEE4",0],["C9","","64E0A4",0],["C10","","3FAE7C",0],["C11","","9AD1BA",0],["C12","","627032",0],["C13","","1A6E3D",0],["C14","","C8E87D",0],["C15","","ACE84C",0],["C16","","305335",0],["C17","","C0ED9C",0],["C18","","9EB33E",0],["C19","","E6ED4F",0],["D1","","ABF8FE",0],["D2","","9EE0F8",0],["D3","","44CDFB",0],["D4","","54A7E9",0],["D5","","ACB7EF",0],["D6","","868DD3",0],["D7","","06ABE3",0],["D8","","3977CC",0],["D9","","0F52BD",0],["D10","","3653AF",0],["D11","","162C7E",0],["D12","","8758A9",0],["D13","","B34EC6",0],["D14","","B37BDC",0],["D15","","301A49",0],["D16","","E3D2FE",0],["D17","","D6BAF5",0],["D18","","2F1E8E",0],["D19","","BCBAE2",0],["D20","","DC99CE",0],["D21","","B5038F",0],["D22","","882893",0],["D23","","E2E4F0",0],["D24","","C7D3F9",0],["D25","","3DBBE3",0],["D26","","CDE7FE",0],["D27","","9A64B8",0],["D28","","1E334E",0],["D29","","21C5C4",0],["D30","","D6FDFC",0],["D31","","1858A2",0],["D32","","02D1F3",0],["D33","","D8C2D9",0],["D34","","9C34AD",0],["D35","","940595",0],["D36","","213244",0],["DH1","","383995",0],["DH10","","CAEDCF",0],["DH11","","1B1213",0],["DH12","","343B1A",0],["DH13","","A5452F",0],["DH14","","BEDDFC",0],["DH15","","26B78E",0],["DH2","","176268",0],["DH3","","DD7D41",0],["DH4","","98A6A6",0],["DH5","","FFE4D6",0],["DH6","","188690",0],["DH7","","0A4241",0],["DH8","","D50527",0],["DH9","","1A70A9",0],["E1","","F6D4CB",0],["E2","","FAF5CD",0],["E3","","FFE4D3",0],["E4","","FCC6AC",0],["E5","","F1C4A5",0],["E6","","DCB387",0],["E7","","98503A",0],["A8","","F6BDE8",0],["E8","","713D2F",0],["A9","","E9639E",0],["E9","","913635",0],["E10","","E4B685",0],["E11","","FEDB99",0],["E12","","DAC898",0],["E13","","FEC993",0],["E14","","8B684C",0],["E15","","874628",0],["E16","","592323",0],["E17","","B2714B",0],["E18","","FDF6EE",0],["E19","","F6F8E3",0],["E20","","F2D8C1",0],["E21","","FAD4BF",0],["E22","","79544E",0],["E23","","F4EFD1",0],["F1","","FBFBFB",0],["F2","","FFFFFF",0],["F3","","B4B4B4",0],["F4","","878787",0],["F5","","464648",0],["F6","","2C2C2C",0],["F7","","010101",0],["F8","","E7D6DC",0],["F9","","ECEAEB",0],["F10","","EFEDEE",0],["F11","","CDCDCD",0],["F12","","CED7D4",0],["G2","","CACADA",0],["G3","","F3F6A9",0],["G4","","FFD785",0],["G5","","FEC832",0],["G6","","E3FBB1",0],["G7","","B2E694",0],["G8","","DEBEE5",0],["G9","","E09DAF",0],["G10","","E84854",0],["G11","","9B9C94",0],["G12","","92AD60",0],["G1","","F8FBE6",0],["G13","","BDCEED",0],["G14","","BDCEE8",0],["G15","","364A89",0],["IC1","","FCFFF8",0],["IC2","","FBF4EC",0],["IC3","","6BB1BB",0],["IC04","","FEFF97",0],["IC5","","E8FAA6",0],["IC6","","F0EEEF",0],["IC7","","F7E3EC",0],["IC8","","FADBF8",0],["IC9","","FDE173",0],["IC10","","F8C0A9",0],["IC11","","F2EEE5",0],["IC12","","F6BBD1",0],["IC13","","FBCBDB",0],["IC14","","FCBF80",0],["IC15","","FD7E77",0],["N5","","3349C3",0],["Q2","","C09DA4",0],["Q3","","937D8A",0],["Q4","","B58B9F",0],["Q5","","BE454A",0],["Q6","","F9D66E",0],["Q7","","907C35",0],["Q8","","B38561",0],["Q9","","E89B7D",0],["Q10","","D07E4A",0],["Q11","","C8E2F9",0],["Q12","","96A09F",0],["Q13","","4E846D",0],["Q1","","D7C6CE",0],["Q14","","768AE1",0],["Q15","","4950C2",0],["R01","","D6C6EB",0],["R02","","F7C3D0",0],["R03","","F2BBC6",0],["R04","","C69495",0],["R05","","EC806D",0],["R06","","EDF878",0],["R07","","FAE393",0],["R08","","E1C9BD",0],["R09","","C6DF5F",0],["R10","","D0E0AF",0],["R11","","9EE5BB",0],["R12","","42ADD1",0],["R13","","7EC5F9",0],["R14","","A9E8E0",0],["R15","","D0DEEF",0],["YX1","","B0AA86",0],["YX10","","A88764",0],["YX11","","BBC6B6",0],["YX12","","909994",0],["YX13","","B0A796",0],["YX14","","AE8082",0],["YX15","","E0D4BC",0],["YX2","","697E80",0],["YX3","","C37463",0],["YX4","","9D7693",0],["YX5","","644B51",0],["YX6","","D0CBAE",0],["YX7","","747D7A",0],["YX8","","C79266",0],["YX9","","C6B2BB",0],["Y1","","FF6FB7",0],["Y2","","FDB583",0],["Y3","","D8FCA4",0],["Y4","","91DAFB",0],["Y5","","E987EA",0],["L2","","F8DA54",0],["L6","","FCFDFF",0],["L7","","56534E",0],["N1","","F13484",0],["N2","","FB852B",0],["N3","","F8ED33",0],["N4","","32C958",0],["P1","","F9F9F9",0],["P2","","ABABAB",0],["P3","","EB903F",0],["P4","","B6DBAF",0],["P5","","FEA2A3",0],["P6","","E79273",0],["P7","","ECDB59",0],["P8","","63CEA2",0],["P9","","F1E9D4",0],["P10","","FFDEE9",0],["P11","","FEBDA7",0],["P12","","E9EDEE",0],["P13","","DBD9DA",0],["P14","","FDC24E",0],["P15","","FCBFD1",0],["P16","","DFAAA4",0],["P17","","ADCBF1",0],["P18","","DBC7EA",0],["P19","","FDA42E",0],["P20","","E8BEC2",0],["P21","","A3656A",0],["P22","","337BAD",0],["P23","","668575",0],["T1","","D40E1F",0],["T2","","945AB1",0],["T3","","1960C8",0],["T4","","1EBA93",0],["T5","","1D779C",0],["T6","","FCECF7",0],["T7","","D8D4D3",0],["S1","","A3E7DC",0],["S2","","78CEE7",0],["S3","","3FCDCE",0],["S4","","7DCA9C",0],["S5","","4E8379",0],["S6","","E3CCBA",0],["S7","","F3C6C0",0],["S8","","D093BC",0],["S9","","D093BC",0],["S10","","9F85CF",0],["S11","","C8E664",0],["S12","","F6BB6F",0],["S13","","A17140",0],["S14","","C76A62",0],["S15","","6B372C",0]]},"panpan289":{"label":"盼盼 Panpan 289","name":"Panpan 289","mm":2.6,"source":"HansBug/pindou-color-data","colors":[["65","","FAF5CD",0],["2","","FCFED6",0],["28","","FCFF92",0],["3","","F7EC5C",0],["74","","F0D83A",0],["29","","FDA951",0],["4","","FA8C4F",0],["88","","FDD94D",0],["90","","F99C5F",0],["89","","F47E36",0],["100","","FEDB99",0],["99","","FDA276",0],["131","","FEC667",0],["138","","F85842",0],["150","","FBF65E",0],["216","","FEFF97",0],["213","","FDE173",0],["223","","FCBF80",0],["218","","FD7E77",0],["242","","F9D66E",0],["276","","FAE393",0],["270","","EDF878",0],["274","","E1C9BD",0],["288","","F3F6A9",0],["289","","FFD785",0],["290","","FEC832",0],["168","","BBC6B6",0],["172","","909994",0],["166","","697E80",0],["167","","E0D4BC",0],["174","","D0CBAE",0],["169","","B0AA86",0],["171","","B0A796",0],["177","","AE8082",0],["170","","A88764",0],["164","","C6B2BB",0],["176","","9D7693",0],["173","","644B51",0],["175","","C79266",0],["165","","C37463",0],["178","","747D7A",0],["69","","FCFDFF",0],["71","","F9F9F9",0],["55","","ABABAB",0],["73","","B6DBAF",0],["72","","FEA2A3",0],["56","","EB903F",0],["157","","63CEA2",0],["159","","E79273",0],["158","","ECDB59",0],["195","","DBD9DA",0],["187","","DBC7EA",0],["185","","F1E9D4",0],["190","","E9EDEE",0],["193","","ADCBF1",0],["183","","337BAD",0],["184","","668575",0],["182","","FDC24E",0],["179","","FDA42E",0],["194","","FEBDA7",0],["186","","FFDEE9",0],["188","","FCBFD1",0],["180","","E8BEC2",0],["189","","DFAAA4",0],["181","","A3656A",0],["67","","D40E1F",0],["24","","F13484",0],["22","","FB852B",0],["21","","F8ED33",0],["23","","32C958",0],["37","","1D779C",0],["68","","1960C8",0],["70","","945AB1",0],["156","","F8DA54",0],["151","","FCECF7",0],["160","","D8D4D3",0],["152","","56534E",0],["231","","A3E7DC",0],["237","","78CEE7",0],["238","","3FCDCE",0],["233","","4E8379",0],["235","","7DCA9C",0],["227","","C8E664",0],["230","","E3CCBA",0],["234","","A17140",0],["226","","6B372C",0],["224","","F6BB6F",0],["228","","F3C6C0",0],["225","","C76A62",0],["229","","D093BC",0],["232","","E58EAE",0],["236","","9F85CF",0],["59","","FF6FB7",0],["60","","FDB583",0],["57","","D8FCA4",0],["58","","91DAFB",0],["61","","E987EA",0],["254","","DAABB3",0],["255","","D6AA87",0],["256","","C1BD8D",0],["257","","96B69F",0],["258","","849DC6",0],["259","","94BFE2",0],["260","","E2A9D2",0],["261","","AB91C0",0],["48","","DFF139",0],["33","","64F343",0],["26","","9FF685",0],["66","","5FDF34",0],["39","","39E158",0],["11","","64E0A4",0],["44","","3EAE7C",0],["10","","1D9B54",0],["79","","2A5037",0],["96","","9AD1BA",0],["97","","627032",0],["106","","1A6E3D",0],["128","","C8E87D",0],["129","","ACE84C",0],["130","","305335",0],["141","","C0ED9C",0],["142","","9EB33E",0],["147","","E6ED4F",0],["191","","26B78E",0],["192","","CAEDCF",0],["207","","176268",0],["206","","0A4241",0],["205","","343B1A",0],["222","","E8FAA6",0],["240","","4E846D",0],["248","","907C35",0],["262","","D0E0AF",0],["269","","9EE5BB",0],["268","","C6DF5F",0],["285","","E3FBB1",0],["286","","B2E694",0],["287","","92AD60",0],["64","","F0FEE4",0],["30","","ABF8FE",0],["63","","9EE0F8",0],["77","","44CDFB",0],["34","","06ABE3",0],["25","","54A7E9",0],["9","","3977CC",0],["52","","0F52BD",0],["42","","3349C3",0],["121","","3DBBE3",0],["122","","2ADED3",0],["120","","1E334E",0],["140","","CDE7FE",0],["139","","D6FDFC",0],["143","","21C5C4",0],["149","","1858A2",0],["163","","02D1F3",0],["196","","213244",0],["202","","18869D",0],["197","","1A70A9",0],["212","","BEDDFC",0],["239","","6BB1BB",0],["263","","C8E2F9",0],["267","","7EC5F9",0],["271","","A9E8E0",0],["265","","42ADD1",0],["279","","D0DEF9",0],["280","","BDCEE8",0],["281","","364A89",0],["46","","ACB7EF",0],["36","","868DD3",0],["8","","3653AF",0],["75","","162C7E",0],["32","","B34EC6",0],["27","","B37BDC",0],["7","","8758A9",0],["94","","E3D2FE",0],["93","","D6BAF5",0],["92","","301A49",0],["105","","BCBAE2",0],["104","","DC99CE",0],["103","","B5038F",0],["102","","882893",0],["101","","2F1E8E",0],["118","","E2E4F0",0],["119","","C7D3F9",0],["124","","9A64B8",0],["153","","D8C2D9",0],["161","","9C34AD",0],["162","","940595",0],["198","","383995",0],["217","","EADBF8",0],["244","","768AE1",0],["249","","4950C2",0],["273","","D6C6EB",0],["18","","F6D4CB",0],["38","","FCC1DD",0],["62","","F6BDE8",0],["6","","E9639E",0],["40","","F1559F",0],["20","","EC4072",0],["41","","C63674",0],["84","","FDDBE9",0],["98","","E575C7",0],["83","","D33997",0],["125","","F7DAD4",0],["126","","F893BF",0],["127","","B5026A",0],["137","","FAD4BF",0],["135","","F5C9CA",0],["221","","FBF4EC",0],["220","","F7E3EC",0],["210","","FBC8DB",0],["215","","F6BBD1",0],["241","","D7C6CE",0],["253","","C09DA4",0],["252","","B58B9F",0],["250","","937D8A",0],["282","","DEBEE5",0],["35","","FF9280",0],["31","","F73D4B",0],["53","","EF4D3E",0],["54","","F92B40",0],["5","","E30328",0],["16","","913635",0],["47","","911932",0],["81","","BB0126",0],["82","","E0677A",0],["116","","874628",0],["117","","592323",0],["136","","F8516D",0],["148","","F45C45",0],["154","","FCADB2",0],["204","","D50527",0],["211","","F8C0A9",0],["245","","E89B7D",0],["246","","D07E4A",0],["243","","BE454A",0],["275","","C69495",0],["266","","F2B8C6",0],["272","","F7C3D0",0],["264","","EC806D",0],["283","","E09DAF",0],["284","","E84854",0],["76","","FFE4D3",0],["49","","FCC6AC",0],["80","","F1C4A5",0],["19","","DCB387",0],["43","","E7B34E",0],["50","","E3A014",0],["17","","985C3A",0],["12","","713D2F",0],["91","","E4B685",0],["87","","DA8C42",0],["112","","DAC898",0],["113","","FEC993",0],["115","","B2714B",0],["114","","8B684C",0],["133","","F6F8E3",0],["134","","F2D8C1",0],["144","","79544E",0],["203","","FFE4D6",0],["208","","DD7D41",0],["199","","A5452F",0],["247","","B38561",0],["15","","FBFBFB",0],["1","","000000",0],["13","","B4B4B4",0],["78","","878787",0],["45","","464648",0],["51","","2C2C2C",0],["14","","010101",0],["85","","E7D6DC",0],["95","","EFEDEE",0],["86","","ECEAEB",0],["123","","CDCDCD",0],["132","","FDF6EE",0],["145","","F4EFD1",0],["146","","CED7D4",0],["201","","98A6A6",0],["200","","1B1213",0],["214","","F0EEEF",0],["219","","FCFFF8",0],["209","","F2EEE5",0],["251","","96A09F",0],["291","","F8FBE6",0],["277","","CACAD2",0],["278","","9B9C94",0]]},"mixiaowo290":{"label":"咪小窝 Mixiaowo 290","name":"Mixiaowo 290","mm":2.6,"source":"HansBug/pindou-color-data","colors":[["77","","FAF5CD",0],["2","","FCFED6",0],["28","","FCFF92",0],["3","","F7EC5C",0],["79","","F0D83A",0],["29","","FDA951",0],["4","","FA8C4F",0],["98","","FDD94D",0],["97","","F99C5F",0],["96","","F47E36",0],["109","","FEDB99",0],["110","","FDA276",0],["116","","FEC667",0],["135","","F85842",0],["150","","FBF65E",0],["216","","FEFF97",0],["213","","FDE173",0],["208","","FCBF80",0],["218","","FD7E77",0],["242","","F9D66E",0],["261","","FAE393",0],["255","","EDF878",0],["259","","E1C9BD",0],["273","","F3F6A9",0],["274","","FFD785",0],["275","","FEC832",0],["168","","BBC6B6",0],["172","","909994",0],["166","","697E80",0],["167","","E0D4BC",0],["159","","D0CBAE",0],["169","","B0AA86",0],["171","","B0A796",0],["162","","AE8082",0],["170","","A88764",0],["164","","C6B2BB",0],["161","","9D7693",0],["173","","644B51",0],["160","","C79266",0],["165","","C37463",0],["163","","747D7A",0],["51","","FCFDFF",0],["62","","F9F9F9",0],["69","","ABABAB",0],["66","","B6DBAF",0],["64","","FEA2A3",0],["63","","EB903F",0],["65","","63CEA2",0],["68","","E79273",0],["67","","ECDB59",0],["178","","DBD9DA",0],["187","","DBC7EA",0],["185","","F1E9D4",0],["190","","E9EDEE",0],["176","","ADCBF1",0],["183","","337BAD",0],["184","","668575",0],["182","","FDC24E",0],["179","","FDA42E",0],["177","","FEBDA7",0],["186","","FFDEE9",0],["180","","FCBFD1",0],["188","","E8BEC2",0],["189","","DFAAA4",0],["181","","A3656A",0],["52","","D40E1F",0],["24","","F13484",0],["22","","FB852B",0],["21","","F8ED33",0],["23","","32C958",0],["55","","1EBA93",0],["37","","1D779C",0],["54","","1960C8",0],["56","","945AB1",0],["53","","F8DA54",0],["151","","FCECF7",0],["157","","D8D4D3",0],["152","","56534E",0],["231","","A3E7DC",0],["224","","78CEE7",0],["225","","3FCDCE",0],["233","","4E8379",0],["222","","7DCA9C",0],["227","","C8E664",0],["230","","E3CCBA",0],["221","","A17140",0],["226","","6B372C",0],["219","","F6BB6F",0],["228","","F3C6C0",0],["220","","C76A62",0],["229","","D093BC",0],["232","","E58EAE",0],["223","","9F85CF",0],["59","","FF6FB7",0],["60","","FDB583",0],["57","","D8FCA4",0],["58","","91DAFB",0],["61","","E987EA",0],["ZG1","","DAABB3",0],["ZG2","","D6AA87",0],["ZG3","","C1BD8D",0],["ZG4","","96B69F",0],["ZG5","","849DC6",0],["ZG6","","94BFE2",0],["ZG7","","E2A9D2",0],["ZG8","","AB91C0",0],["48","","DFF139",0],["33","","64F343",0],["26","","9FF685",0],["78","","5FDF34",0],["39","","39E158",0],["11","","64E0A4",0],["44","","3EAE7C",0],["10","","1D9B54",0],["84","","2A5037",0],["100","","9AD1BA",0],["99","","627032",0],["111","","1A6E3D",0],["119","","C8E87D",0],["117","","ACE84C",0],["122","","305335",0],["133","","C0ED9C",0],["141","","9EB33E",0],["147","","E6ED4F",0],["174","","26B78E",0],["175","","CAEDCF",0],["194","","176268",0],["193","","0A4241",0],["192","","343B1A",0],["207","","E8FAA6",0],["240","","4E846D",0],["248","","907C35",0],["262","","D0E0AF",0],["254","","9EE5BB",0],["253","","C6DF5F",0],["270","","E3FBB1",0],["271","","B2E694",0],["272","","92AD60",0],["76","","F0FEE4",0],["30","","ABF8FE",0],["75","","9EE0F8",0],["82","","44CDFB",0],["34","","06ABE3",0],["25","","54A7E9",0],["9","","3977CC",0],["71","","0F52BD",0],["42","","3349C3",0],["130","","3DBBE3",0],["113","","2ADED3",0],["120","","1E334E",0],["142","","CDE7FE",0],["136","","D6FDFC",0],["132","","21C5C4",0],["149","","1858A2",0],["156","","02D1F3",0],["196","","213244",0],["202","","18869D",0],["197","","1A70A9",0],["212","","BEDDFC",0],["239","","6BB1BB",0],["263","","C8E2F9",0],["252","","7EC5F9",0],["256","","A9E8E0",0],["250","","42ADD1",0],["264","","D0DEF9",0],["265","","BDCEE8",0],["266","","364A89",0],["46","","ACB7EF",0],["36","","868DD3",0],["8","","3653AF",0],["80","","162C7E",0],["32","","B34EC6",0],["27","","B37BDC",0],["7","","8758A9",0],["89","","E3D2FE",0],["90","","D6BAF5",0],["91","","301A49",0],["104","","BCBAE2",0],["105","","DC99CE",0],["106","","B5038F",0],["107","","882893",0],["108","","2F1E8E",0],["126","","E2E4F0",0],["128","","C7D3F9",0],["125","","9A64B8",0],["153","","D8C2D9",0],["155","","9C34AD",0],["158","","940595",0],["198","","383995",0],["217","","EADBF8",0],["244","","768AE1",0],["234","","4950C2",0],["258","","D6C6EB",0],["18","","F6D4CB",0],["38","","FCC1DD",0],["74","","F6BDE8",0],["6","","E9639E",0],["40","","F1559F",0],["20","","EC4072",0],["41","","C63674",0],["103","","FDDBE9",0],["95","","E575C7",0],["94","","D33997",0],["131","","F7DAD4",0],["112","","F893BF",0],["124","","B5026A",0],["140","","FAD4BF",0],["139","","F5C9CA",0],["206","","FBF4EC",0],["205","","F7E3EC",0],["210","","FBC8DB",0],["215","","F6BBD1",0],["241","","D7C6CE",0],["238","","C09DA4",0],["237","","B58B9F",0],["235","","937D8A",0],["267","","DEBEE5",0],["35","","FF9280",0],["31","","F73D4B",0],["72","","EF4D3E",0],["73","","F92B40",0],["5","","E30328",0],["16","","913635",0],["47","","911932",0],["92","","BB0126",0],["93","","E0677A",0],["115","","874628",0],["129","","592323",0],["134","","F8516D",0],["148","","F45C45",0],["154","","FCADB2",0],["191","","D50527",0],["211","","F8C0A9",0],["245","","E89B7D",0],["246","","D07E4A",0],["243","","BE454A",0],["260","","C69495",0],["251","","F2B8C6",0],["257","","F7C3D0",0],["249","","EC806D",0],["268","","E09DAF",0],["269","","E84854",0],["81","","FFE4D3",0],["49","","FCC6AC",0],["85","","F1C4A5",0],["19","","DCB387",0],["43","","E7B34E",0],["50","","E3A014",0],["17","","985C3A",0],["12","","713D2F",0],["102","","E4B685",0],["101","","DA8C42",0],["118","","DAC898",0],["127","","FEC993",0],["114","","B2714B",0],["123","","8B684C",0],["143","","F6F8E3",0],["138","","F2D8C1",0],["137","","79544E",0],["203","","FFE4D6",0],["195","","DD7D41",0],["199","","A5452F",0],["247","","B38561",0],["15","","FBFBFB",0],["1","","FFFFFF",0],["13","","B4B4B4",0],["83","","878787",0],["45","","464648",0],["70","","2C2C2C",0],["14","","010101",0],["86","","E7D6DC",0],["87","","EFEDEE",0],["88","","ECEAEB",0],["121","","CDCDCD",0],["144","","FDF6EE",0],["146","","F4EFD1",0],["145","","CED7D4",0],["201","","98A6A6",0],["200","","1B1213",0],["214","","F0EEEF",0],["204","","FCFFF8",0],["209","","F2EEE5",0],["236","","96A09F",0],["276","","F8FBE6",0],["277","","CACAD2",0],["278","","9B9C94",0]]},"artkalc197":{"label":"优肯 Artkal C 197 (2.6 mm)","name":"Artkal C 197","mm":2.6,"source":"HansBug/pindou-color-data","colors":[["C01","","FFFFFF",0],["C02","","000000",0],["C03","","FF8200",0],["C04","","FF7F41",0],["C05","","E4002B",0],["C06","","BA0C2F",0],["C07","","EF95CF",0],["C08","","F04E98",0],["C09","","E24585",0],["C10","","F8E08E",0],["C11","","FFC72C",0],["C12","","9BE198",0],["C13","","93C90E",0],["C14","","009A44",0],["C15","","007862",0],["C16","","FF6A39",0],["C17","","FF5C39",0],["C18","","8DC8E8",0],["C19","","56B7E6",0],["C20","","0090DA",0],["C21","","004EA8",0],["C22","","F7CED7",0],["C23","","EAA794",0],["C24","","EFD19F",0],["C25","","A388D8",0],["C26","","9569BE",0],["C27","","483698",0],["C28","","C67D30",0],["C29","","A4493D",0],["C30","","93262C",0],["C31","","A37F74",0],["C32","","683431",0],["C33","","A7A8AA",0],["C34","","888B8D",0],["C35","","A09F9D",0],["C36","","E68699",0],["C37","","0067B9",0],["C38","","418FDE",0],["C39","","A4DBE8",0],["C40","","E1D555",0],["C41","","FBDB65",0],["C42","","FFC845",0],["C43","","A50034",0],["C44","","FF808B",0],["C45","","4A7C68",0],["C46","","F0EC74",0],["C47","","ECC3B2",0],["C48","","FFB549",0],["C49","","EC86D0",0],["C50","","C5B4E3",0],["C51","","F5E1A4",0],["C52","","5949A7",0],["C53","","C4D600",0],["C54","","00B2A9",0],["C55","","6CC24A",0],["C56","","97999B",0],["C57","","D50032",0],["C58","","2C2D65",0],["C59","","861F41",0],["C60","","BBDDE6",0],["C61","","F1E6B2",0],["C62","","FF585D",0],["C63","","F1A7DC",0],["C64","","CE0058",0],["C65","","E3775E",0],["C66","","F67599",0],["C67","","B14EB5",0],["C68","","62B5E5",0],["C69","","333F48",0],["C70","","183028",0],["C71","","FFA300",0],["C72","","FFB25B",0],["C73","","B86125",0],["C74","","D5A286",0],["C75","","AA8066",0],["C76","","FF6D6A",0],["C77","","B04A5A",0],["C78","","862633",0],["C79","","99D6EA",0],["C80","","00968F",0],["C81","","48A9C5",0],["C82","","009CBD",0],["C83","","0085AD",0],["C84","","ABAD23",0],["C85","","999B30",0],["C86","","008522",0],["C87","","D9D9D6",0],["C88","","C8C9C7",0],["C89","","B1B3B3",0],["C90","","53565A",0],["C91","","D69A2D",0],["C92","","F1C6A7",0],["C93","","DCA59C",0],["C94","","F18070",0],["C95","","FA9370",0],["C96","","EA7600",0],["C97","","6CACE4",0],["C98","","71C5E8",0],["C99","","63B1BC",0],["C100","","00A3E1",0],["C101","","0076CF",0],["C102","","0057B8",0],["C103","","007D9B",0],["C104","","007D8A",0],["C105","","006A8E",0],["C106","","007096",0],["C107","","D5CB9F",0],["C108","","CFC493",0],["C109","","C0B561",0],["C110","","998542",0],["C111","","897630",0],["C112","","CAC7A7",0],["C113","","BFBB98",0],["C114","","B0AA7E",0],["C115","","899064",0],["C116","","7A7256",0],["C117","","5B792F",0],["C118","","9CDBD9",0],["C119","","7AE1BF",0],["C120","","77C19A",0],["C121","","00B467",0],["C122","","37C400",0],["C123","","257226",0],["C124","","008675",0],["C125","","007367",0],["C126","","DFC8E7",0],["C127","","D48BC8",0],["C128","","EABEDB",0],["C129","","DD74A1",0],["C130","","C1A0DA",0],["C131","","FB637E",0],["C132","","C800A1",0],["C133","","87189D",0],["C134","","AAB7E8",0],["C135","","B9B7E7",0],["C136","","7BA2E1",0],["C137","","5F8DDA",0],["C138","","5576D1",0],["C139","","485CC7",0],["C140","","476ACD",0],["C141","","365ABD",0],["C142","","3448AA",0],["C143","","002D72",0],["C144","","E9E186",0],["C145","","A6192E",0],["C146","","86C8BC",0],["C147","","866761",0],["C148","","EFDBB2",0],["C149","","8B84D7",0],["C150","","3E4827",0],["C151","","C0AEE4",0],["C152","","C2BDE0",0],["C153","","F3CFB3",0],["C154","","C58B68",0],["C155","","63666A",0],["C156","","4B4F54",0],["C157","","1D252D",0],["CE01","","D1DDE6",0],["CE02","","9BB8D3",0],["CE03","","6AD1E3",0],["CE04","","03C1AA",0],["CE05","","2CD5C4",0],["CE06","","147BD1",0],["CE07","","BFA5B8",0],["CE08","","9B7793",0],["CE09","","693C5E",0],["CE10","","C964CF",0],["CE11","","AD1AAC",0],["CE12","","FFC27B",0],["CE13","","E6A65D",0],["CE14","","D38235",0],["CE15","","C16C18",0],["CE16","","F5B6CD",0],["CE17","","5D2A2C",0],["CG01","","B2B9BF",1],["CG02","","BEA6BA",1],["CG03","","62B6D8",1],["CG04","","8985C1",1],["CG05","","CB9D96",1],["CG06","","8ABA96",1],["CG07","","94637F",1],["CP01","","8ECAB2",1],["CP02","","CD8F75",1],["CP03","","CBA16C",1],["CP04","","BC8DA4",1],["CP05","","8A506A",1],["CP06","","3063A3",1],["CP07","","3E5694",1],["CT01","","C6C6D280",1],["CT02","","E664A680",1],["CT03","","F6875C80",1],["CT04","","C4C76480",1],["CT05","","32B19580",1],["CT06","","757DD180",1],["CT07","","4EB1DC80",1],["CT08","","5688A080",1],["CT09","","6B849D80",1]]},"artkalm221":{"label":"优肯 Artkal M 221 (MARD-compatible)","name":"Artkal M 221","mm":2.6,"source":"HansBug/pindou-color-data","colors":[["MA1","","FFF6D4",0],["MA2","","F6F9E5",0],["MA3","","FFFBAA",0],["MA4","","FFDF58",0],["MA5","","FECF4D",0],["MA6","","FFA80C",0],["MA7","","FF8837",0],["MA8","","EAC431",0],["MA9","","F1AA8F",0],["MA10","","FF8F42",0],["MA11","","FFE1A1",0],["MA12","","F4B39C",0],["MA13","","FFB100",0],["MA14","","FF6510",0],["MA15","","DAF05C",0],["MA16","","F9FBBE",0],["MA17","","FFD976",0],["MA18","","FFCA9A",0],["MA19","","EC7A92",0],["MA20","","E5BE55",0],["MA21","","FFE596",0],["MA22","","F7E898",0],["MA23","","F0CBB1",0],["MA24","","F5FCD1",0],["MA25","","F6D487",0],["MA26","","FFCB4B",0],["MB1","","D2E318",0],["MB2","","79CD41",0],["MB3","","82D7A1",0],["MB4","","65DF4F",0],["MB5","","5FC873",0],["MB6","","49D1AE",0],["MB7","","009696",0],["MB8","","08774F",0],["MB9","","183823",0],["MB10","","83CFC3",0],["MB11","","5A6A27",0],["MB12","","045F45",0],["MB13","","E2FFB3",0],["MB14","","9DD12E",0],["MB15","","254B3C",0],["MB16","","D0FCAD",0],["MB17","","8CA12A",0],["MB18","","D2D958",0],["MB19","","49BCA9",0],["MB20","","E8FDEC",0],["MB21","","188B81",0],["MB22","","0C5C5B",0],["MB23","","3D461B",0],["MB24","","EAFCB6",0],["MB25","","538771",0],["MB26","","8A7A40",0],["MB27","","D2DEB6",0],["MB28","","9FF4C2",0],["MB29","","BBD747",0],["MB30","","F5FFE6",0],["MB31","","BFE1C2",0],["MB32","","9FBA5C",0],["MC1","","D5E3DE",0],["MC2","","BBF1F4",0],["MC3","","73C0DF",0],["MC4","","33B3E1",0],["MC5","","00A3CA",0],["MC6","","58A0D9",0],["MC7","","0588CC",0],["MC8","","005F9E",0],["MC9","","086FB9",0],["MC10","","52B4E0",0],["MC11","","00A9B9",0],["MC12","","1C375A",0],["MC13","","CEE0F0",0],["MC14","","EBF5F4",0],["MC15","","00AAAD",0],["MC16","","004C7D",0],["MC17","","5BD4F6",0],["MC18","","36515C",0],["MC19","","0F909D",0],["MC20","","0084B7",0],["MC21","","DDECFA",0],["MC22","","84BAC2",0],["MC23","","CBDBDB",0],["MC24","","88BDE2",0],["MC25","","B3ECE0",0],["MC26","","3E9EBF",0],["MC27","","E9F0F3",0],["MC28","","C2CDE3",0],["MC29","","586B8F",0],["MD1","","7292E2",0],["MD2","","6E8CCC",0],["MD3","","13419A",0],["MD4","","1E3175",0],["MD5","","B850B2",0],["MD6","","AD92E9",0],["MD7","","5F2D91",0],["MD8","","D7CDF1",0],["MD9","","BDB2E6",0],["MD10","","251B58",0],["MD11","","A2BBE7",0],["MD12","","CF8BBE",0],["MD13","","B80096",0],["MD14","","A137A7",0],["MD15","","4A388A",0],["MD16","","DCE8F6",0],["MD17","","AECEF0",0],["MD18","","B38EDB",0],["MD19","","E8D2EA",0],["MD20","","B460C3",0],["MD21","","842A94",0],["MD22","","4B5E9E",0],["MD23","","D6D2E2",0],["MD24","","7776D1",0],["MD25","","3837A8",0],["MD26","","DDC8DD",0],["ME1","","FDD6C9",0],["ME2","","FFD8F4",0],["ME3","","FFA0C3",0],["ME4","","F67AAB",0],["ME5","","DA5B95",0],["ME6","","FF4B78",0],["ME7","","9E156B",0],["ME8","","FFE1E6",0],["ME9","","F378CA",0],["ME10","","BE316D",0],["ME11","","FFE8DA",0],["ME12","","FFA5C6",0],["ME13","","B53883",0],["ME14","","FFDDC7",0],["ME15","","E5C6D0",0],["ME16","","F9F2EB",0],["ME17","","E7DCE6",0],["ME18","","FFD3E6",0],["ME19","","FFD5EC",0],["ME20","","F3DCE8",0],["ME21","","BD9CA3",0],["ME22","","C578A5",0],["ME23","","A37A9B",0],["ME24","","F4E8FF",0],["MF1","","FF7F67",0],["MF2","","FF5D46",0],["MF3","","E3564A",0],["MF4","","C7270D",0],["MF5","","BE0A27",0],["MF6","","83411E",0],["MF7","","730037",0],["MF8","","AB0022",0],["MF9","","DE7995",0],["MF10","","AB613C",0],["MF11","","5C2929",0],["MF12","","F34461",0],["MF13","","C64239",0],["MF14","","FFA5BB",0],["MF15","","BC0018",0],["MF16","","FFD7CB",0],["MF17","","F19D8A",0],["MF18","","DC7B45",0],["MF19","","C74651",0],["MF20","","D8A89E",0],["MF21","","EB9DB9",0],["MF22","","FFD4D4",0],["MF23","","EA8474",0],["MF24","","F7C1C5",0],["MF25","","FC5768",0],["MG1","","FFE7C8",0],["MG2","","FFD6C3",0],["MG3","","EEB694",0],["MG4","","D9A58C",0],["MG5","","F49630",0],["MG6","","F19937",0],["MG7","","AC745B",0],["MG8","","694A42",0],["MG9","","F7CC9D",0],["MG10","","C7832F",0],["MG11","","D3C391",0],["MG12","","E6B986",0],["MG13","","C67B4F",0],["MG14","","8D6242",0],["MG15","","F6F7E2",0],["MG16","","F2E7D7",0],["MG17","","786562",0],["MG18","","FEF5EB",0],["MG19","","F0AB47",0],["MG20","","AE6949",0],["MG21","","BF8C6E",0],["MH1","","FFFFFF00",0],["MH2","","FFFFFF",0],["MH3","","9B9B9B",0],["MH4","","6D6D6D",0],["MH5","","4D4D4D",0],["MH6","","3A3A3A",0],["MH7","","000000",0],["MH8","","FFF0F1",0],["MH9","","E3E6DF",0],["MH10","","E2E0E7",0],["MH11","","BCB8B5",0],["MH12","","F7F3E1",0],["MH13","","EADDC8",0],["MH14","","B8C8C5",0],["MH15","","99ABAC",0],["MH16","","444236",0],["MH17","","FAFBF5",0],["MH18","","FEFEEE",0],["MH19","","FBF2E4",0],["MH20","","9EAEAD",0],["MH21","","FDFFF2",0],["MH22","","EDEDE7",0],["MH23","","B4B8A7",0],["MM1","","AEBBA8",0],["MM2","","729376",0],["MM3","","6A8890",0],["MM4","","BEB39D",0],["MM5","","B8B78C",0],["MM6","","B2AC93",0],["MM7","","BC9F94",0],["MM8","","968381",0],["MM9","","B39D88",0],["MM10","","B796A3",0],["MM11","","B096B9",0],["MM12","","685951",0],["MM13","","C79582",0],["MM14","","CA694F",0],["MM15","","929FA9",0]]},"artkals":{"label":"Artkal S (5 mm)","name":"Artkal S","mm":5.0,"source":"maxcleme/beadcolors","colors":[["S01","White","EAEEF3",0],["S02","Burning Sand","EE927C",0],["S03","Tangerine","FFA630",0],["S04","Orange","EB6027",0],["S05","Tall Poppy","CB3531",0],["S06","Raspberry Pink","EF67B2",0],["S07","Gray","959698",0],["S08","Emerald","1FC467",0],["S09","Dark Green","00685E",0],["S10","Baby Blue","2EABD8",0],["S11","Dark Blue","004FA4",0],["S12","Pastel Lavender","9165B2",0],["S13","Black","292A2B",0],["S14","Sandstorm","E1C835",0],["S15","Redwood","9A4541",0],["S16","Brown","65463D",0],["S17","Light Brown","895D49",0],["S18","Sand","DCA384",0],["S19","Bubble Gun","F2BFB8",0],["S20","Green","009053",0],["S21","Pastel Green","8BB23A",0],["S22","Purple","48337E",0],["S23","Royal Purple","644FA4",0],["S24","True Blue","0078BF",0],["S25","Hot Pink","DC519A",0],["S26","Magenta","DA4383",0],["S27","Yellow","EAC125",0],["S28","Lily Pink","DDA1CC",0],["S29","Pastel Yellow","E1D367",0],["S30","Shadow Green","9EC9CD",0],["S31","Sea Mist","B2D7CE",0],["S32","Beeswax","EED39E",0],["S33","Maverick","C0B7D7",0],["S34","Red","B61927",0],["S35","Mona Lisa","E1CCD2",0],["S36","Old Pink","CE6D83",0],["S37","Blue-Green","60BFCB",0],["S38","Burgundy","A3134A",0],["S39","Yellow Orange","E68739",0],["S40","Carnation Pink","E182B0",0],["S41","Copper","9D5B2E",0],["S42","Silver","676B73",0],["S43","Dark Gray","70757B",0],["S44","Sky Blue","A7CDDE",0],["S45","Medium Turquoise","00A5A1",0],["S46","Bright Green","8EC324",0],["S47","Marigold","B27938",0],["S48","Corn","ECC03D",0],["S49","Mulberry Wood","921E5D",0],["S50","Mandys Pink","E2A488",0],["S51","Spring Sun","DFDABD",0],["S52","Picasso","EADE7F",0],["S53","Turquoise","55A4D9",0],["S54","Light Blue","0084CE",0],["S55","Pistachio","97CF87",0],["S56","Bright Carrot","F96F40",0],["S57","Buccaneer","B35540",0],["S58","Paprika","B11836",0],["S59","Butterfly Bush","4F3989",0],["S60","Lavender","8A7EC2",0],["S61","Key Lomen Pie","CDC03F",0],["S62","Green Tea","00765F",0],["S63","Metallic Gold","696046",1],["S64","Black Rock","36384D",0],["S65","Canary","E2E65D",0],["S66","Blaze Orange","E5533C",0],["S67","Vanilla","E9C1A6",0],["S68","Tan","CC9B74",0],["S69","Mine Shaft","383E44",0],["S70","Drark Algae","ADAD29",0],["S71","Jade Green","007D2B",0],["S72","Light Sea Blue","68C4D2",0],["S73","Steel Blue","5AB0BF",0],["S74","Azure","009EC2",0],["S75","Dark Steel Blue","0084B2",0],["S76","Sea Blue","0093A9",0],["S77","Ghost While","D4D8D3",0],["S78","Ash Grey","C2C4C2",0],["S79","Light Gray","A7ACAD",0],["S80","Dark Olive","8F8E3C",0],["S81","Deer","BF9168",0],["S82","Clay","AA744E",0],["S83","Sienna","BB6833",0],["S84","Deep Chestnut","BE5D65",0],["S85","Red Wine","99323A",0],["S86","Goldenrod","E8AE00",0],["S87","Coral Red","EC625E",0],["S88","Dark Pink","D62779",0],["S89","Charcoal Gray","565A5E",0],["S90","Pastel Orange","D9B35E",0],["S91","Brunswick Green","153838",0],["S92","Dandelion","CEA433",0],["S93","Pale Skin","DCB794",0],["S94","Warm Blush","DD9285",0],["S95","Salmon","E07B69",0],["S96","Apricot","EF7F61",0],["S97","Papaya","DC772B",0],["S98","Himalaya Blue","6AAEDB",0],["S99","Waterfall","61BBD3",0],["S100","Lagoon","279BBE",0],["S101","Electric Blue","00A7E3",0],["S102","Pool Blue","0077CA",0],["S103","Caribbian Blue","005AA9",0],["S104","Deep Water","007F9E",0],["S105","Petrol Blue","007D91",0],["S106","Wegdewood Blue","00649A",0],["S107","Pond Blue","006C9F",0],["S108","Seashell Beige","CFC179",0],["S109","Beige","C4AE64",0],["S110","Beach Beige","AB9745",0],["S111","Caffe Latté","978138",0],["S112","Oaktree Brown","907C41",0],["S113","Khaki","B6AE84",0],["S114","Light Greengray","A59F65",0],["S115","Mossy Green","938D54",0],["S116","Earth Green","8D8B51",0],["S117","Sage Green","7F7E49",0],["S118","Pinetree Green","5B6E35",0],["S119","Frosty Blue","8AD5C9",0],["S120","Polar Mint","7CD2A5",0],["S121","Celadon Green","72AC9A",0],["S122","Eucalyptus","00B26F",0],["S123","Clover Field","3EB724",0],["S124","Pooltable Felt","0D7535",0],["S125","Snake Green","007D6E",0],["S126","Dark Eucalyptus","006E69",0],["S127","Marsmallow Rose","DFC3E1",0],["S128","Light Grape","D38ED4",0],["S129","Rosebud Pink","D5A6BA",0],["S130","Fuschia","D6668E",0],["S131","Candy Violet","B8AAD9",0],["S132","Flamingo","DF486D",0],["S133","Pink Plum","BC3CA6",0],["S134","Amethyst","803897",0],["S135","Moonlight Blue","A7BAE1",0],["S136","Summer Rain","AFB8DF",0],["S137","Azur Blue","6B9AD4",0],["S138","Cornflower Blue","5A89CE",0],["S139","Forget Me Not","658AD0",0],["S140","Indigo","566CBD",0],["S141","Horizon Blue","4D74C6",0],["S142","Cobolt","416DBE",0],["S143","Royal Blue","30429E",0],["S144","Marine","024288",0],["S145","Pale Yellow Moss","D6CA6A",0],["S146","Bloodrose Red","9D1A38",0],["S147","Spearmint","80B7A1",0],["S148","Mocha","7A594F",0],["S149","Creme","EFDBA1",0],["S150","Iris Violet","8884D0",0],["S151","Forrest Green","345621",0],["S152","Lilac","AEADDC",0],["S153","Pale Lilac","BCC3E1",0],["S154","Sahara Sand","E3C09A",0],["S155","Sunkissed Teint","C58B60",0],["S156","Steel Grey","5A5F65",0],["S157","Iron Grey","4C5156",0],["S158","Pepper","3A3E42",0],["S159","Oslo Gray","82878B",0],["SE1","Moon","F1F4FA",0],["SE2","Raincloud","A2B2D3",0],["SE3","Winter","83A6DC",0],["SE4","Tide","70D8ED",0],["SE5","Aquamarine","69E3D5",0],["SE6","Neptune","2A63C8",0],["SE7","Rose Quartz","D5AEC9",0],["SE8","Mauve","AF739A",0],["SE9","Dark Mauve","704B6D",0],["SE10","Wild Berry","D977CD",0],["SE11","Sugar Plum","B643AC",0],["SE12","Desert","DEB181",0],["SE13","Peanut","BC9D77",0],["SE14","Toffee","AC7F5D",0],["SE15","Bark","A37E66",0],["SG1","Glow Yellow","C6C0B4",1],["SG2","Glow Pink","DFABBC",1],["SG3","Glow Blue","6FADC1",1],["SL1","Glitter Blue","3687B2",1],["SL2","Glitter Green","117C52",1],["SL3","Glitter Red","C22933",1],["SL4","Glitter Yellow","CDBD00",1],["SN1","Neon Orange","FF7405",1],["SN2","Neon Green","01C63E",1],["SN3","Neon Yellow","F2F33C",1],["SN4","Neon Rose","FF477A",1],["SP1","Pearlescent Green","93CA87",1],["SP2","Pearlescent Tangerine","FE7F54",1],["SP3","Pearlescent Orange","F0A149",1],["SP4","Pearlescent Pink","DC9EA9",1],["SP5","Pearlescent Red","D63C53",1],["SP6","Pearlescent Blue","2C4E97",1],["SP7","Pearlescent Purple","64427E",1],["SP8","Pearlescent White","E0DBCA",1],["ST1","Transparent","C7BCB8",1],["ST2","Transparent Pink","F95498",1],["ST3","Transparent Tangering","FF7C45",1],["ST4","Transparent Yellow","DADF3D",1],["ST5","Transparent Green","0FAE58",1],["ST6","Transparent Purple","8477B3",1]]},"perler":{"label":"Perler (5 mm)","name":"Perler","mm":5.0,"source":"maxcleme/beadcolors","colors":[["80-15179","Evergreen","305545",0],["80-15181","Light Grey","B3BAB8",0],["80-15182","Lavender","AF9FCE",0],["80-15199","Shamrock","008F53",0],["80-15200","Cobalt","0065B1",0],["80-15201","Midnight","2F3C55",0],["80-15202","Robin's Egg","A9CDD5",0],["80-15203","Flamingo","F2AFB7",0],["80-15204","Salmon","E1747A",0],["80-15205","Fawn","C9A385",0],["80-15206","Pewter","94A19D",0],["80-15207","Charcoal","4F595A",0],["80-15208","Toasted Marshmallow","DEDACE",0],["80-15210","Orchid","B1628E",0],["80-15211","Tomato","D14337",0],["80-15212","Spice","D9593A",0],["80-15213","Apricot","F5A168",0],["80-15214","Sherbet","D8E47C",0],["80-15215","Mist","93B0BD",0],["80-15216","Sky","4AC0D8",0],["80-15217","Lagoon","00A4AC",0],["80-15218","Teal","047F8A",0],["80-15219","Fern","7F971A",0],["80-15220","Olive","696E31",0],["80-15961","Cherry","9D2B3A",0],["80-19001","White","EAEFEE",0],["80-19002","Creme","E1E2BB",0],["80-19003","Yellow","E7CE3E",0],["80-19004","Orange","EB7B31",0],["80-19005","Red","B0353C",0],["80-19006","Bubblegum","D8729A",0],["80-19007","Purple","684B86",0],["80-19008","Dark Blue","0E5092",0],["80-19009","Light Blue","278CC9",0],["80-19010","Dark Green","007B4E",0],["80-19011","Light Green","18C7B1",0],["80-19012","Brown","674C44",0],["80-19017","Grey","909497",0],["80-19018","Black","323234",0],["80-19020","Rust","995043",0],["80-19021","Light Brown","936848",0],["80-19033","Peach","E9BFB9",0],["80-19035","Tan","C5AC90",0],["80-19038","Magenta","E04284",0],["80-19052","Pastel Blue","4A9CCF",0],["80-19053","Pastel Green","6DCC94",0],["80-19054","Pastel Lavender","937FBF",0],["80-19056","Pastel Yellow","E9E290",0],["80-19057","Cheddar","FBB146",0],["80-19058","Toothpaste","96D1D4",0],["80-19059","Hot Coral","DD595B",0],["80-19060","Plum","A75D9D",0],["80-19061","Kiwi Lime","69B845",0],["80-19062","Turquoise","0098C5",0],["80-19063","Blush","F99297",0],["80-19070","Periwinkle Blue","6683B7",0],["80-19079","Light Pink","E1BCCE",0],["80-19080","Green","4DAB64",0],["80-19083","Pink","D45496",0],["80-19088","Raspberry","983864",0],["80-19090","Butterscotch","DA9964",0],["80-19091","Parrot Green","009188",0],["80-19092","Dark Grey","585C61",0],["80-19093","Blueberry Creme","85A8E3",0],["80-19096","Cranapple","843947",0],["80-19097","Prickly Pear","BBC938",0],["80-19098","Sand","E5BE9E",0],["80-15240","Mint","B3EED5",0],["80-15241","Sour Apple","A3DE6F",0],["80-15242","Cotton Candy","F479B0",0],["80-15243","Grape","503B9C",0],["80-15244","Rose","D25D72",0],["80-15245","Iris","4E56A3",0],["80-15246","Tangerine","FD5918",0],["80-15247","Forest","005D57",0],["80-15248","Eggplant","6F3255",0],["80-15249","Honey","DA8C2C",0],["80-15250","Gingerbread","7E5446",0],["80-15251","Thistle","8C8CA7",0],["80-15252","Slate Blue","5E6D7B",0],["80-15253","Denim","4C6388",0],["80-15254","Sage","9AA98E",0],["80-15255","Orange Cream","EFB79B",0],["80-15256","Fruit Punch","CA3B65",0],["80-15257","Fuchsia","CB59B9",0],["80-15258","Mulberry","714875",0],["80-15259","Slime","C8C85C",0],["80-15260","Stone","988C8C",0],["80-15261","Dark Spruce","14313B",0],["80-15262","Cocoa","392928",0],["80-15265","Twilight Plum","C685B1",0],["80-15266","Caribbean Sea","6CC8AD",0],["80-15267","Frosted Lilac","CDB7C3",0],["80-15273","Brick","FC9574",0],["80-15274","Rich Butter","F6CA69",0],["80-15275","Peacock","0090AC",0],["80-15276","Carnation Pink","F8C7C9",0],["80-15089","Neon Blue","406AE1",1],["80-15268","Sunflower","DEBA0B",0],["80-15269","Lemon","F6D901",0],["80-15263","Celery","BED4A6",0],["80-15239","Mocha","C8B693",0],["80-15272","Coral","FF9A8B",0]]},"perlermini":{"label":"Perler Mini","name":"Perler Mini","mm":2.6,"source":"maxcleme/beadcolors","colors":[["80-15179","Evergreen","305545",0],["80-15181","Light Grey","B3BAB8",0],["80-19001","White","EAEFEE",0],["80-19002","Creme","E1E2BB",0],["80-19003","Yellow","E7CE3E",0],["80-19004","Orange","EB7B31",0],["80-19005","Red","B0353C",0],["80-19006","Bubblegum","D8729A",0],["80-19007","Purple","684B86",0],["80-19008","Dark Blue","0E5092",0],["80-19009","Light Blue","278CC9",0],["80-19011","Light Green","18C7B1",0],["80-19012","Brown","674C44",0],["80-19017","Grey","909497",0],["80-19018","Black","323234",0],["80-19021","Light Brown","936848",0],["80-19033","Peach","E9BFB9",0],["80-19035","Tan","C5AC90",0],["80-19010","Dark Green","007B4E",0],["80-19052","Pastel Blue","4A9CCF",0],["80-19054","Pastel Lavender","937FBF",0],["80-19056","Pastel Yellow","E9E290",0],["80-19057","Cheddar","FBB146",0],["80-19058","Toothpaste","96D1D4",0],["80-19059","Hot Coral","DD595B",0],["80-19060","Plum","A75D9D",0],["80-19061","Kiwi Lime","69B845",0],["80-19062","Turquoise","0098C5",0],["80-19063","Blush","F99297",0],["80-19079","Light Pink","E1BCCE",0],["80-19080","Green","4DAB64",0],["80-19083","Pink","D45496",0],["80-19088","Raspberry","983864",0],["80-19090","Butterscotch","DA9964",0],["80-19091","Parrot Green","009188",0],["80-19092","Dark Grey","585C61",0],["80-19093","Blueberry Creme","85A8E3",0],["80-19096","Cranapple","843947",0],["80-19097","Prickly Pear","BBC938",0],["80-19098","Sand","E5BE9E",0],["80-14110","Toasted Marshmallow","DEDACE",0]]},"hama":{"label":"Hama Midi (5 mm)","name":"Hama Midi","mm":5.0,"source":"maxcleme/beadcolors","colors":[["H01","White","E5ECF1",0],["H02","Cream","E4E4C5",0],["H03","Yellow","E9C704",0],["H04","Orange","D14803",0],["H05","Red","B4060E",0],["H06","Pink","EA8AA5",0],["H07","Purple","712297",0],["H08","Blue","0239A3",0],["H09","Light Blue","025BC3",0],["H10","Green","027643",0],["H11","Light Green","19CDA7",0],["H12","Brown","3E271A",0],["H13","Transparent Red","C02435",1],["H14","Transparent Yellow","E4AA32",1],["H16","Transparent Green","37B876",1],["H17","Grey","838F98",0],["H18","Black","141315",0],["H19","Clear","D8D2CE",1],["H20","Reddish Brown","8D2A0F",0],["H21","Light Brown","BE6C21",0],["H22","Dark Red","91020A",0],["H24","Translucent Purple","683E9A",1],["H25","Translucent Brown","87593D",1],["H26","Matt Rose","E8A498",0],["H27","Beige","DCB18E",0],["H28","Dark Green","1E2C1C",0],["H29","Claret","BF0142",0],["H30","Burgundy","4E0C1B",0],["H31","Turquoise","489AB9",0],["H32","Neon Fuchsia","FF208D",1],["H33","Cerise","FF3956",0],["H34","Neon Yellow","E5EF13",1],["H35","Neon Red","FF2833",1],["H36","Neon Blue","2353B0",1],["H37","Neon Green","06B73C",1],["H38","Neon Orange","FD8600",1],["H39","Fluorescent Yellow","F1F21C",1],["H40","Fluorescent Orange","FE630B",1],["H41","Fluorescent Blue","2659B2",1],["H42","Fluorescent Green","0CBD51",1],["H43","Pastel Yellow","E7E45A",0],["H44","Pastel Red","F96160",0],["H45","Pastel Purple","8E69CD",0],["H46","Pastel Blue","51AEE4",0],["H47","Pastel Green","80DF96",0],["H48","Pastel Pink","D67AD1",0],["H49","Azure","0FACD1",0],["H60","Teddybear Brown","F0981E",0],["H70","Light Grey","A5B3C0",0],["H71","Dark Grey","445059",0],["H75","Tan","B78C6D",0],["H76","Nougat","8A5937",0],["H77","Cloudy White","CED1C8",0],["H78","Light Peach","F7C1AA",0],["H79","Apricot","F87633",0],["H82","Plum","91175A",0],["H83","Petrol Blue","037A9F",0],["H84","Olive Green","687836",0],["H95","Pastel Rose","DD9BA3",0],["H96","Pastel Lilac","B491AD",0],["H97","Pastel Ice Blue","8AAFC2",0],["H98","Pastel Mint","94CCA4",0],["H15","Transparent Blue","487ED5",1],["H55","Green (Glow in the Dark)","FAF8ED",1],["H56","Red (Glow in the Dark)","EDBF9F",1],["H57","Blue (Glow in the Dark)","C4D0E3",1],["H61","Gold","D99350",0],["H62","Silver","48474A",0],["H63","Bronze","42312F",0],["H64","Pearl","EFEBE4",1],["H72","Translucent Pink","F097B0",1],["H73","Translucent Aqua","59AEF5",1],["H74","Translucent Lilac","5B55BD",1],["H101","Eucalyptus","A9C39B",0],["H102","Forest Green","356B2D",0],["H103","Light Yellow","FFE660",0],["H104","Lime","BCD122",0],["H105","Light Apricot","FFAC78",0],["H106","Light Lavender","CCC5ED",0],["H107","Lavender","6A87C1",0],["H108","Aubergine","2A2536",0],["H109","Cloudy Grey","8A847F",0],["H110","Matcha","838956",0],["H111","Dark Blush","835854",0],["H112","Blush","AD8A82",0],["H113","Aqua","5F887B",0],["H114","Cherry Red","9A2C31",0],["H115","Bright Green","6E975F",0],["H116","Midnight Blue","222838",0],["H117","Taupe Grey","777169",0],["H118","Maroon Red","612932",0],["H119","Sky Blue","4167B4",0]]},"hamamini":{"label":"Hama Mini (2.5 mm)","name":"Hama Mini","mm":2.5,"source":"maxcleme/beadcolors","colors":[["H01","White","E5ECF1",0],["H02","Cream","E4E4C5",0],["H03","Yellow","E9C704",0],["H04","Orange","D14803",0],["H05","Red","B4060E",0],["H06","Pink","EA8AA5",0],["H07","Purple","712297",0],["H08","Blue","0239A3",0],["H09","Light Blue","025BC3",0],["H10","Green","027643",0],["H11","Light Green","19CDA7",0],["H12","Brown","3E271A",0],["H13","Transparent Red","C02435",1],["H14","Transparent Yellow","E4AA32",1],["H15","Transparent Blue","487ED5",1],["H16","Transparent Green","37B876",1],["H17","Grey","838F98",0],["H18","Black","141315",0],["H19","Clear","D8D2CE",1],["H20","Reddish Brown","8D2A0F",0],["H21","Light Brown","BE6C21",0],["H22","Dark Red","91020A",0],["H24","Translucent Purple","683E9A",1],["H26","Matt Rose","E8A498",0],["H27","Beige","DCB18E",0],["H28","Dark Green","1E2C1C",0],["H29","Claret","BF0142",0],["H30","Burgundy","4E0C1B",0],["H31","Turquoise","489AB9",0],["H32","Neon Fuchsia","FF208D",1],["H33","Cerise","FF3956",0],["H34","Neon Yellow","E5EF13",1],["H35","Neon Red","FF2833",1],["H36","Neon Blue","2353B0",1],["H37","Neon Green","06B73C",1],["H38","Neon Orange","FD8600",1],["H42","Fluorescent Green","0CBD51",1],["H43","Pastel Yellow","E7E45A",0],["H44","Pastel Red","F96160",0],["H45","Pastel Purple","8E69CD",0],["H46","Pastel Blue","51AEE4",0],["H47","Pastel Green","80DF96",0],["H48","Pastel Pink","D67AD1",0],["H49","Azure","0FACD1",0],["H60","Teddybear Brown","F0981E",0],["H70","Light Grey","A5B3C0",0],["H71","Dark Grey","445059",0],["H75","Tan","B78C6D",0],["H76","Nougat","8A5937",0],["H77","Cloudy White","CED1C8",0],["H78","Light Peach","F7C1AA",0],["H79","Apricot","F87633",0],["H82","Plum","91175A",0],["H83","Petrol Blue","037A9F",0],["H84","Olive Green","687836",0],["H95","Pastel Rose","DD9BA3",0],["H96","Pastel Lilac","B491AD",0],["H97","Pastel Ice Blue","8AAFC2",0],["H98","Pastel Mint","94CCA4",0],["H101","Eucalyptus","A9C39B",0],["H102","Forest Green","356B2D",0],["H103","Light Yellow","FFE660",0],["H104","Lime","BCD122",0],["H105","Light Apricot","FFAC78",0],["H106","Light Lavender","CCC5ED",0],["H107","Lavender","6A87C1",0],["H108","Aubergine","2A2536",0],["H109","Cloudy Grey","8A847F",0],["H110","Matcha","838956",0],["H111","Dark Blush","835854",0],["H112","Blush","AD8A82",0],["H113","Aqua","5F887B",0],["H114","Cherry Red","9A2C31",0],["H115","Bright Green","6E975F",0],["H116","Midnight Blue","222838",0],["H117","Taupe Grey","777169",0],["H118","Maroon Red","612932",0],["H119","Sky Blue","4167B4",0]]}}'''

if __name__ == "__main__":
    sys.exit(main())
