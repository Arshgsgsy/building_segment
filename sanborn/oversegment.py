"""Heuristic over-segmenter (guide §9.3): cut a sheet into too many regions along every ink line.

Defaults come from the `oversegment:` block in config.yaml, then the sheet's city entry in
`oversegment_by_city:` (city from data/manifest.csv); any flag overrides them for one run.

examples:
  uv run python -m sanborn.oversegment data/synthetic/synthetic_sheet.png --gt data/synthetic/synthetic_gt.geojson
  uv run python -m sanborn.oversegment data/raw/<sheet>.jp2 --crop 2000,3000,1500,1500 --debug
  uv run python -m sanborn.oversegment data/raw/<sheet>.jp2 --ink-contrast 15 --min-ink-support 0.4 --tag c15

Outputs go to outputs/overseg/<sheet>[_crop...][_tag]/: qa.png, regions.geojson, labels.png, stats.json,
and with --debug also input.png, normalized.png, ink.png, fill.png, bridges.png (bridged wall gaps in red).

regions.geojson properties: id, area_px, fill_frac, flag ("round" / "dark" / "pale": likely not a
building, the labeling tool steps through them), suspect (1: a wall runs inside the region, check for a
missed cut).
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from sanborn import ROOT, load_config, manifest
from sanborn.data import load_sheet, normalize, paper_color
from sanborn.evaluate import overseg_checks
from sanborn.labels import load_polygons, polygonize, qa_overlay, rasterize, write_geojson

# Table 15: what each knob does
HELP = {
    "ink_contrast": "ink if this many L* units darker than its surroundings. Lower: faded walls missed, buildings merge. Raise: fill texture or stains read as ink, buildings get holes",
    "ink_width": "lines thinner than this count as ink, px. Raise: thick party walls missed. Lower: narrow colored strips read as ink",
    "ink_L_dark": "anything darker than this L* is ink regardless of surroundings (solid black symbols)",
    "min_ink_cc": "drop ink blobs smaller than this, px. Raise: letters still punch holes. Lower: short wall pieces disappear",
    "ink_close": "closing size bridging gaps in walls, px (0 = off). Raise: buildings merge through wall gaps. Lower: narrow gaps between buildings close up",
    "fill_chroma": "a pixel is colored fill if chroma above this. Raise: paper or stains become regions. Lower: pale fills break up or vanish",
    "fill_smooth": "majority-filter size for the fill mask, px (0 = off). Raise: pale fills come out speckled. Lower: thin fills between walls vanish",
    "min_area": "keep regions of at least this many px. Raise: noise regions appear. Lower: tiny outbuildings are lost",
    "min_ink_support": "keep a region if at least this fraction of its outline lies on ink. Raise: stains or colored symbols kept. Lower: buildings with open or dashed sides dropped",
    "grow": "grow regions this far into the surrounding ink and halo, px; also the ink-support distance. Raise: thick walls stay unassigned. Lower: polygons spill onto streets at open sides",
    "max_hole": "fill enclosed unlabeled pockets up to this size, px. Raise: large symbols leave holes. Lower: small courtyards get filled",
    "dash_support": "dashed-outline rule: min share of the outline on any ink, dashes included. Lower: dashed porches/sheds missed. Raise: text blobs on yellowed paper kept",
    "dash_chroma": "dashed-outline rule: min chroma above the nearby paper. Lower: very pale sheds missed. Raise: paper blobs or stains kept",
    "dash_L": "dashed-outline rule: min mean L* (rejects printed numbers)",
    "dash_rect": "dashed-outline rule: min area / bounding-rectangle area. Lower: odd-shaped dashed porches missed. Raise: border strips kept",
    "bridge_length": "wall-gap bridging: min straight wall piece to extend, px. Lower: more cuts from short strokes (labels, hatching). Raise: short wall pieces aren't extended, merges stay",
    "bridge_gap": "wall-gap bridging: widest gap to bridge, px. Raise: wide openings still merge buildings. Lower: false cuts across courts and yards",
    "bridge_span": "wall-gap bridging: only walls in ink networks at least this long, px (skips text and symbols)",
    "sliver_r": "drop non-compact regions narrower than about 2x this, px. Raise: strips along walls kept. Lower: narrow sheds dropped",
    "sliver_circ": "slivers must also be less compact than this (4*pi*A/P^2). Lower: X-hatch triangles and small sheds dropped",
    "hue_cut": "split a region whose fill holds two hues at least this many degrees apart, along the color change (0 = off). Lower: shading and stains split one fill. Raise: neighbours of close colors (yellow / green, ~25 degrees apart) stay merged",
    "hue_chroma": "hue cut: only fill of at least this chroma counts toward a hue, since the hue of near-gray fill is noise. Lower: pale or gray fills split by noise. Raise: pale neighbours stay merged",
    "dash_cut": "also cut along dashed lines at least this long, px (0 = off). Lower: short dashed strokes and text cut. Raise: dashed party walls and shed sides stay merged",
    "simplify": "Douglas-Peucker tolerance for the output polygons, px",
}


def params(sheet=None, **overrides):
    """Over-segmenter parameters from config.yaml, with overrides.

    With `sheet` (a sheet id or image path), the `oversegment_by_city` entry for that sheet's city in the
    manifest is applied on top of the shared values. Without it (the wall fit in `dissolve`), only the
    shared values are used, so label outlines follow one convention on every sheet.
    """
    return argparse.Namespace(**{**load_config()["oversegment"], **city_knobs(sheet), **overrides})


def city_knobs(sheet):
    """The city's knob overrides for `sheet`, from config.yaml's `oversegment_by_city`; {} if none."""
    if sheet is None:
        return {}
    by_city = load_config().get("oversegment_by_city") or {}
    row = next((r for r in manifest.read() if r["sheet_id"] == Path(sheet).stem), {})
    knobs = by_city.get(row.get("city", "").lower()) or {}
    unknown = set(knobs) - set(load_config()["oversegment"])
    if unknown:
        raise SystemExit(f"oversegment_by_city.{row['city']}: unknown knobs {sorted(unknown)}")
    return knobs


def segment(rgb, p, paper=None, sheet_edges=(True, True, True, True)):
    """Steps 1-6 of §9.3. Returns the region map (int32, 0 = not a building) and debug arrays.

    Departures from the guide, all found on real LOC sheets: ink is local contrast instead of an
    absolute L* threshold; regions are built from colored fill (kept by ink support) instead of from
    all non-ink pixels (kept by fill fraction), so buildings can't leak into the streets; dashed
    outlines get their own keep rule (4b); regions are split along bridged wall gaps (4c) and,
    optionally, along fill hue changes and dashed lines (4d); slivers are dropped (7).

    `sheet_edges` says which sides (top, bottom, left, right) of `rgb` are real sheet edges;
    for a crop, regions cut by the crop border are kept.
    """
    # 1-3a. paper normalization, ink and colored fill
    m = ink_masks(rgb, p, paper)
    norm, L, chroma, ink, ink_all, fill = (m[k] for k in ("normalized", "L", "chroma", "ink", "ink_all", "fill"))
    paper = m["paper"]
    del m

    # 3. regions: 4-connected components of colored fill that isn't ink. Paper is not part of any
    # region, so a building can't leak into the streets through a door, light court or dashed line.
    n, regions = cv2.connectedComponents((fill & ~ink).view(np.uint8), connectivity=4)
    regions = fill_pockets(regions, p.max_hole)  # text and noise holes inside a fill region

    # 4. keep regions that are big enough, outlined mostly by ink (stains aren't), and off the sheet edge
    area = np.bincount(regions.ravel(), minlength=n)
    support = ink_support(regions, ink, p.grow)
    keep = (area >= p.min_area) & (support >= p.min_ink_support)
    edge = np.zeros(n, bool)
    sides = [regions[0], regions[-1], regions[:, 0], regions[:, -1]]
    for is_edge, side in zip(sheet_edges, sides):
        if is_edge:
            edge[np.unique(side)] = True
    keep &= ~edge
    keep[0] = False
    # 4b. dashed outlines: their dashes are text-sized, so step 2 removed them and support is low.
    # Measure support against all ink instead, and require a pale, roughly rectangular fill that
    # stands out in chroma from the paper around it (text on yellowed paper also has high support).
    cand = ~keep & ~edge & (area >= p.min_area)
    cand[0] = False
    if cand.any():
        cand &= ink_support(regions, ink_all, p.grow) >= p.dash_support
        objs = ndimage.find_objects(regions)
        for i in np.flatnonzero(cand):
            keep[i] = dashed_box(regions, i, objs[i - 1], L, chroma, p)
    # step 4d's cuts, while the small ink blobs (dashes, text) are still at hand
    dashes = dash_cut_lines(ink_all, ink, p.dash_cut) if p.dash_cut > 0 else None
    hue = hue_field(norm, fill & ~ink_all, p.hue_chroma) if p.hue_cut > 0 else None
    del ink_all
    ids = np.zeros(n, np.int32)
    ids[keep] = np.arange(1, keep.sum() + 1)
    labels = ids[regions]
    del regions

    # 4c. split along walls whose drawn gaps (doors, openings, broken party walls) no fill-based
    # region can respect: bridge gaps up to bridge_gap px that continue a straight wall, then cut
    # along ink plus bridges (ink too, since step 3's pocket fill painted over free-floating walls)
    bridges = bridge_walls(ink, p.bridge_length, p.bridge_gap, p.bridge_span)
    cut = bridges | ink
    # 4d (optional). Neighbours of another construction class (pink brick, yellow frame, green, ...)
    # are often drawn against each other with only a dashed line or no line, so the fill connects
    # them: also cut along long dashed lines, then split regions that hold two fill hues
    if dashes is not None:
        cut |= dashes
    labels = split_pieces(labels, cut, p.min_area)
    del cut
    hue_cuts = None
    if hue is not None:
        labels, hue_cuts = split_by_hue(labels, hue, p.hue_cut, p.min_area)
        del hue

    # 5. grow into the surrounding ink and anti-aliased halo, nearest region first, so party walls
    # split down the middle
    labels = grow(labels, p.grow)
    # 6. fill small unlabeled pockets enclosed by one region (text, symbols)
    labels = fill_pockets(labels, p.max_hole)
    # 7. drop slivers: strips under ~2*sliver_r px wide that aren't compact (strips along walls,
    # scale-bar pieces, compass spokes). The smallest real sheds are ~25 px wide.
    labels = drop_slivers(labels, p.sliver_r, p.sliver_circ)
    return labels, {"normalized": norm, "ink": ink, "fill": fill, "paper": paper, "bridges": bridges,
                    "L": L, "chroma": chroma, "hue_cuts": hue_cuts, "dashes": dashes}


def hue_field(norm, ok, min_chroma, sigma=2.0, chunk=1024):
    """Step 4d: per pixel, the hue (a*b* angle, degrees) of the fill around it, NaN where there is no
    fill nearby of chroma `min_chroma` (the hue of near-gray fill is noise). a* and b* are averaged
    over the fill only (`ok`: text, walls and dashes left out) with a Gaussian of `sigma` px, so text
    and dashes take the color of the fill around them."""
    H, W = ok.shape
    out = np.full((H, W), np.nan, np.float32)
    pad = int(4 * sigma) + 2
    for y0 in range(0, H, chunk):  # in row chunks: a full sheet in float32 is ~200 MB per array
        a0, a1 = max(y0 - pad, 0), min(y0 + chunk + pad, H)
        lab = cv2.cvtColor(norm[a0:a1], cv2.COLOR_RGB2LAB)
        w = ok[a0:a1].astype(np.float32)
        a = cv2.GaussianBlur(lab[..., 1] * w, (0, 0), sigma)
        b = cv2.GaussianBlur(lab[..., 2] * w, (0, 0), sigma)
        del lab
        w = cv2.GaussianBlur(w, (0, 0), sigma)
        h = np.degrees(np.arctan2(b, a))
        h[(w < 0.15) | (np.hypot(a, b) < min_chroma * w)] = np.nan
        r0 = y0 - a0
        out[y0:min(y0 + chunk, H)] = h[r0:r0 + min(chunk, H - y0)]
    return out


def split_by_hue(labels, hue, min_deg, min_area, passes=2, iters=10):
    """Step 4d: split regions holding two fill hues at least `min_deg` degrees apart.

    Per region, a 2-means on the hue (`hue_field`) of its pixels; if both group centers are `min_deg`
    apart and each group holds `min_area` px, every pixel joins its nearer group (pixels without a hue
    the group of the nearest pixel with one) and the region is cut where the group changes. The cut
    is a partition boundary, so it is closed even where text or a dashed line lies on the color change.
    `passes` rounds, so a region of three colors splits twice. Returns the labels and the cut pixels."""
    cos_max = np.cos(np.radians(min_deg))
    cuts = np.zeros(labels.shape, bool)
    for _ in range(passes):
        cut = np.zeros(labels.shape, bool)
        for i, sl in enumerate(ndimage.find_objects(labels), 1):
            if sl is None:
                continue
            m = labels[sl] == i
            h = hue[sl]
            known = m & ~np.isnan(h)
            if known.sum() < 2 * min_area:
                continue
            v = np.exp(1j * np.radians(h[known]))
            group = two_means(v, iters)
            if group is None:
                continue
            g, c = group
            if (c[0] * np.conj(c[1])).real > cos_max or min(np.bincount(g, minlength=2)) < min_area:
                continue
            cls = np.full(m.shape, -1, np.int8)
            cls[known] = g
            # pixels without a hue (text far from fill, gray fill): the group of the nearest one with
            _, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
            cls = np.where(m, cls[iy, ix], -1)
            # the boundary of the partition: a pixel whose right or lower neighbour is in the other group
            b = np.zeros(m.shape, bool)
            b[:, :-1] |= (cls[:, :-1] != cls[:, 1:]) & m[:, :-1] & m[:, 1:]
            b[:-1] |= (cls[:-1] != cls[1:]) & m[:-1] & m[1:]
            cut[sl] |= b
        if not cut.any():
            break
        labels = split_pieces(labels, cut, min_area)
        cuts |= cut
    return labels, cuts


def two_means(v, iters):
    """2-means of unit complex numbers `v` (hue directions): (group per item, the two unit centers),
    or None if all items are alike. Starts from the 10th and 90th percentile of the angle around the
    mean, so a few stray pixels (halo, stain specks) can't claim a center."""
    mean = v.mean()
    if abs(mean) < 1e-9:
        return None
    mean /= abs(mean)
    lo, hi = np.percentile(np.angle(v * np.conj(mean)), [10, 90])
    if hi - lo < 1e-6:
        return None
    c = mean * np.exp(1j * np.array([lo, hi]))
    for _ in range(iters):
        g = ((v * np.conj(c[1])).real > (v * np.conj(c[0])).real).astype(np.int8)
        s = np.array([v[g == 0].sum(), v[g == 1].sum()])
        if (np.abs(s) < 1e-9).any():
            return None
        new = s / np.abs(s)
        if np.allclose(new, c):
            break
        c = new
    return g, c


def dash_cut_lines(ink_all, ink, min_len):
    """Step 4d: dashed lines at least `min_len` px long (sanborn.labels.dash_lines, as in the wall fit)."""
    from sanborn.labels import dash_lines
    return dash_lines(ink_all, ink, min_len=int(min_len))


def ink_masks(rgb, p, paper=None):
    """Steps 1-3a of §9.3, shared with the dissolve-time wall fit (sanborn.labels.wall_masks).

    Returns the paper color, the paper-normalized sheet, L* and chroma (float32), `ink` (walls: text-sized
    blobs removed, small gaps closed), `ink_all` (also the small blobs: dashes, text) and `fill` (colored).
    """
    # 1. paper normalization
    paper = paper_color(rgb) if paper is None else paper
    norm = normalize(rgb, paper)
    lab = cv2.cvtColor(norm, cv2.COLOR_RGB2LAB)
    L, chroma = lab[..., 0], np.hypot(lab[..., 1], lab[..., 2])
    del lab

    # 2. ink: thin lines darker than their surroundings (black-hat on L*), or very dark. Local
    # contrast works on light and dark paper alike and ignores broad dull fills, where an absolute
    # L* threshold can't do both. Then drop text-sized blobs and close small wall gaps. `ink_all`
    # keeps the small blobs too: dashed outlines are made of them.
    kernel = np.ones((int(p.ink_width),) * 2, np.uint8)
    ink_all = (cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, kernel) > p.ink_contrast) | (L < p.ink_L_dark)
    ink = remove_small(ink_all, p.min_ink_cc)
    k = int(p.ink_close)
    if k > 0:
        close = np.ones((k, k), np.uint8)
        ink = cv2.morphologyEx(ink.view(np.uint8), cv2.MORPH_CLOSE, close).view(bool)
        ink_all = cv2.morphologyEx(ink_all.view(np.uint8), cv2.MORPH_CLOSE, close).view(bool)

    # 3a. colored fill
    fill = (chroma > p.fill_chroma).view(np.uint8)
    if p.fill_smooth > 1:
        fill = cv2.medianBlur(fill, int(p.fill_smooth) | 1)  # majority vote removes fill speckle
    return {"paper": paper, "normalized": norm, "L": L, "chroma": chroma, "ink": ink, "ink_all": ink_all,
            "fill": fill.view(bool)}


def ink_support(regions, ink, dist):
    """Per region, the fraction of its boundary pixels within `dist` px of ink.

    Crop borders don't count as boundary (edge padding), so buildings cut by a crop keep their support.
    """
    pad = np.pad(regions, 1, mode="edge")
    c = pad[1:-1, 1:-1]
    edge = (c != pad[:-2, 1:-1]) | (c != pad[2:, 1:-1]) | (c != pad[1:-1, :-2]) | (c != pad[1:-1, 2:])
    edge &= regions > 0
    near = cv2.dilate(ink.view(np.uint8), np.ones((2 * dist + 1,) * 2, np.uint8)).view(bool)
    n = regions.max() + 1
    total = np.bincount(regions[edge], minlength=n)
    return np.bincount(regions[edge & near], minlength=n) / np.maximum(total, 1)


def remove_small(mask, min_px):
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.view(np.uint8), connectivity=8)
    big = stats[:, cv2.CC_STAT_AREA] >= min_px
    big[0] = False
    return big[cc]


def grow(labels, r):
    if r <= 0:
        return labels
    dist, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    take = dist <= r
    labels = labels.copy()
    labels[take] = labels[iy[take], ix[take]]
    return labels


def fill_pockets(labels, max_hole):
    n, pockets, stats, _ = cv2.connectedComponentsWithStats((labels == 0).view(np.uint8), connectivity=4)
    H, W = labels.shape
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for i in np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] <= max_hole) + 1:
        x, y, w, h = stats[i, :4]
        if x == 0 or y == 0 or x + w == W or y + h == H:
            continue
        sl = np.s_[y - 1:y + h + 1, x - 1:x + w + 1]
        m = pockets[sl] == i
        ring = cv2.dilate(m.view(np.uint8), cross).view(bool) & ~m
        owners = np.unique(labels[sl][ring])
        if len(owners) == 1:
            labels[sl][m] = owners[0]
    return labels


def dashed_box(regions, i, sl, L, chroma, p, pad=25):
    """Step 4b test for region i: pale, stands out in chroma from nearby paper, roughly rectangular."""
    H, W = regions.shape
    y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)
    x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)
    m = (regions[y0:y1, x0:x1] == i).view(np.uint8)
    inside = m.view(bool)
    C, Lw = chroma[y0:y1, x0:x1], L[y0:y1, x0:x1]
    if Lw[inside].mean() < p.dash_L:  # dark: printed numbers, not fills
        return False
    near = cv2.dilate(m, np.ones((11, 11), np.uint8)).view(bool)
    far = cv2.dilate(m, np.ones((41, 41), np.uint8)).view(bool)
    ring = far & ~near & (Lw > 70)  # paper around the fill, ink excluded
    if ring.sum() < 20 or C[inside].mean() - np.percentile(C[ring], 25) < p.dash_chroma:
        return False
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, (w, h), _ = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    return inside.sum() / max(w * h, 1) >= p.dash_rect


def straight_pieces(ink, length):
    """Horizontal and vertical straight ink pieces at least `length` px long.

    An even `length` is rounded up to odd: OpenCV centers an even kernel off by one, so the opening
    came out shifted 1 px off the ink and wall-gap bridging silently stopped working."""
    length = int(length) | 1
    u = ink.view(np.uint8)
    h = cv2.morphologyEx(u, cv2.MORPH_OPEN, np.ones((1, length), np.uint8)).view(bool)
    v = cv2.morphologyEx(u, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8)).view(bool)
    return h, v


def wall_network(ink, span):
    """Ink in 8-connected components at least `span` px long: walls, not text or symbols."""
    n, cc, st, _ = cv2.connectedComponentsWithStats(ink.view(np.uint8), connectivity=8)
    big = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]) >= span
    big[0] = False
    return big[cc]


def bridge_walls(ink, length, gap, span, iters=2):
    """Pixels that bridge gaps of <= `gap` px continuing a straight wall piece (>= `length` px, part of
    the wall network) along its own row or column to the next straight ink piece."""
    h, v = straight_pieces(ink, length)
    line = h | v
    added = np.zeros_like(ink)
    cur = ink
    for _ in range(iters):
        net = wall_network(cur, span)
        a = _bridge_rows(cur, h & ~v & net, line, gap)  # h & ~v: walls minus their junctions
        a |= _bridge_rows(cur.T.copy(), (v & ~h & net).T.copy(), line.T.copy(), gap).T
        a &= ~cur
        if not a.any():
            break
        added |= a
        cur = cur | a
    return added


def _bridge_rows(ink, start, end, gap, chunk=512):
    """Fill row runs of non-ink no longer than `gap` with a `start` pixel at one end and an `end` pixel at the other."""
    H, W = ink.shape
    out = np.zeros_like(ink)
    ar = np.arange(W, dtype=np.int32)
    for y0 in range(0, H, chunk):
        sl = slice(y0, min(y0 + chunk, H))
        k = ink[sl]
        left = np.maximum.accumulate(np.where(k, ar, -1), axis=1)
        right = np.minimum.accumulate(np.where(k, ar, W)[:, ::-1], axis=1)[:, ::-1]
        ry, rx = np.nonzero(~k & (left >= 0) & (right < W) & (right - left - 1 <= gap))
        l, r = left[ry, rx], right[ry, rx]
        s, e = start[sl], end[sl]
        good = (s[ry, l] & e[ry, r]) | (s[ry, r] & e[ry, l])
        out[sl][ry[good], rx[good]] = True
    return out


def split_pieces(labels, cut, min_area):
    """Split each region along `cut` into 4-connected pieces. A piece under `min_area` rejoins the
    neighbouring piece of the same region it touches most (or stays, if the whole region is small)."""
    m = (labels > 0) & ~cut
    n, pieces = cv2.connectedComponents(m.view(np.uint8), connectivity=4)
    area = np.bincount(pieces.ravel(), minlength=n)
    parent = np.zeros(n, np.int32)
    parent[pieces[m]] = labels[m]
    small = np.flatnonzero(area < min_area)
    small = small[small > 0]
    objs = ndimage.find_objects(pieces)
    H, W = labels.shape
    for i in small[np.argsort(area[small])]:
        sl = objs[i - 1]
        y0, y1 = max(sl[0].start - 6, 0), min(sl[0].stop + 6, H)
        x0, x1 = max(sl[1].start - 6, 0), min(sl[1].stop + 6, W)
        pc = pieces[y0:y1, x0:x1]
        me = pc == i
        ring = cv2.dilate(me.view(np.uint8), np.ones((9, 9), np.uint8)).view(bool) & ~me
        nb = pc[ring]
        nb = nb[(nb > 0) & (nb != i) & (parent[nb] == parent[i])]
        if len(nb):
            j = np.bincount(nb).argmax()
            pc[me] = j
            area[j] += area[i]
    used = np.unique(pieces[m])
    remap = np.zeros(n, np.int32)
    remap[used] = np.arange(1, len(used) + 1)
    out = remap[pieces]
    out[~m] = 0  # bridge and ink pixels; grow gives them back
    return out


def region_shape(labels):
    """Per region id: inscribed radius, circularity 4*pi*A/P^2 of its largest outline, area, solidity."""
    n = int(labels.max())
    pad = np.pad(labels, 1, mode="edge")
    c = pad[1:-1, 1:-1]
    edge = (c != pad[:-2, 1:-1]) | (c != pad[2:, 1:-1]) | (c != pad[1:-1, :-2]) | (c != pad[1:-1, 2:])
    dt = cv2.distanceTransform(((labels > 0) & ~edge).view(np.uint8), cv2.DIST_L2, 5) + 1
    r = np.zeros(n + 1)
    r[1:] = ndimage.maximum(dt, labels, np.arange(1, n + 1))
    area = np.bincount(labels.ravel(), minlength=n + 1)
    perim, hull = np.ones(n + 1), np.ones(n + 1)
    for i, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None:
            continue
        contours, _ = cv2.findContours(np.pad((labels[sl] == i).view(np.uint8), 1),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        big = max(contours, key=cv2.contourArea)
        perim[i] = max(cv2.arcLength(big, True), 1)
        hull[i] = max(cv2.contourArea(cv2.convexHull(big)), 1)
    return r, 4 * np.pi * area / perim ** 2, area, np.minimum(area / hull, 1)


def drop_slivers(labels, r_max, circ_max):
    r, circ, _, _ = region_shape(labels)
    bad = (r <= r_max) & (circ < circ_max)
    bad[0] = False
    ids = np.zeros(len(r), np.int32)
    ids[~bad] = np.arange((~bad).sum())
    return ids[labels]


def region_flags(labels, dbg):
    """Per region id: `flag` marks likely non-buildings for a bulk delete in QGIS (never dropped here):
    "round" (isolated circles: fire-alarm symbols, tanks, cisterns), "dark" (binding-margin marks, black
    symbols), "pale" (isolated near-gray blobs: text on yellowed paper). `suspect` marks regions with
    a straight wall inside them, likely two buildings merged through a gap too wide to bridge."""
    n = int(labels.max())
    r, circ, area, solidity = region_shape(labels)
    idx = np.arange(1, n + 1)
    mean_L = np.zeros(n + 1)
    mean_L[1:] = ndimage.mean(dbg["L"], labels, idx)
    mean_C = np.zeros(n + 1)
    mean_C[1:] = ndimage.mean(dbg["chroma"], labels, idx)
    has_nbr = np.zeros(n + 1, bool)
    for a, b in ((labels[:, 1:], labels[:, :-1]), (labels[1:], labels[:-1])):
        touch = (a > 0) & (b > 0) & (a != b)
        has_nbr[a[touch]] = has_nbr[b[touch]] = True
    flag = np.full(n + 1, "", object)
    flag[(mean_C < 8) & ~has_nbr] = "pale"
    flag[mean_L < 33] = "dark"
    # area / inscribed circle: 1 for a circle, 1.17-1.20 for dashed (ragged) cisterns, >= 1.27 for any square
    flag[~has_nbr & (r >= 20) & (area / (np.pi * np.maximum(r, 1) ** 2) <= 1.22) & (solidity >= 0.95)] = "round"
    # straight wall-network ink lying fully inside one region (a 9x9 window holding only its label)
    h, v = straight_pieces(dbg["ink"], 31)
    walls = (h | v) & wall_network(dbg["ink"], 150)
    lf = labels.astype(np.float32)
    k = np.ones((9, 9), np.uint8)
    lo = cv2.erode(lf, k)
    inner = walls & (cv2.dilate(lf, k) == lo) & (lo > 0)
    wall_len = np.bincount(labels[inner], minlength=n + 1) / 3.0  # ~3 px thick walls
    return {i: {"flag": flag[i], "suspect": int(wall_len[i] >= 60)} for i in idx}


def main():
    defaults = load_config()["oversegment"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet", help="JP2/PNG/JPG sheet image")
    ap.add_argument("--crop", help="x,y,w,h in sheet pixels; runs in about a second instead of ~15 s")
    ap.add_argument("--gt", help="label GeoJSON in pixel coords, or a labeled sheet's data/labels/<sheet> folder "
                    "(scored inside its extents only), to score the §9.2 rules against")
    ap.add_argument("--debug", action="store_true", help="also save input, normalized, ink and fill images")
    ap.add_argument("--tag", help="suffix for the output folder so runs don't overwrite each other")
    tuning = ap.add_argument_group("tuning (defaults from config.yaml; the sheet's city entry in oversegment_by_city wins)")
    for k, v in defaults.items():
        tuning.add_argument("--" + k.replace("_", "-"), type=number, default=None, metavar="N",
                            help=f"{HELP.get(k, '')} [default {v}]")
    a = ap.parse_args()
    p = params(a.sheet, **{k: getattr(a, k) for k in defaults if getattr(a, k) is not None})
    city = city_knobs(a.sheet)
    if city:
        print(f"city knobs: {city}")

    sheet = load_sheet(a.sheet)
    H, W = sheet.shape[:2]
    paper = paper_color(sheet)
    x, y, w, h = map(int, a.crop.split(",")) if a.crop else (0, 0, W, H)
    w, h = min(w, W - x), min(h, H - y)
    rgb = np.ascontiguousarray(sheet[y:y + h, x:x + w])

    t = time.time()
    labels, dbg = segment(rgb, p, paper, (y == 0, y + h == H, x == 0, x + w == W))
    seconds = time.time() - t

    name = Path(a.sheet).stem + (f"_crop{x}_{y}_{w}_{h}" if a.crop else "") + (f"_{a.tag}" if a.tag else "")
    out = ROOT / "outputs" / "overseg" / name
    out.mkdir(parents=True, exist_ok=True)

    n = int(labels.max())
    area = np.bincount(labels.ravel(), minlength=n + 1)
    fill_frac = np.bincount(labels.ravel(), weights=dbg["fill"].ravel(), minlength=n + 1) / np.maximum(area, 1)
    flags = region_flags(labels, dbg)
    props = {i: {"area_px": int(area[i]), "fill_frac": round(float(fill_frac[i]), 3), **flags[i]}
             for i in range(1, n + 1)}
    write_geojson(out / "regions.geojson", polygonize(labels, p.simplify, (x, y)), props,
                  {"sheet": Path(a.sheet).name, "params": vars(p)})
    cv2.imwrite(str(out / "labels.png"), labels.astype(np.uint16))
    cv2.imwrite(str(out / "qa.png"), cv2.cvtColor(qa_overlay(rgb, labels), cv2.COLOR_RGB2BGR))
    if a.debug:
        cv2.imwrite(str(out / "input.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "normalized.png"), cv2.cvtColor((dbg["normalized"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "ink.png"), np.where(dbg["ink"], 0, 255).astype(np.uint8))
        cv2.imwrite(str(out / "fill.png"), np.where(dbg["fill"], 0, 255).astype(np.uint8))
        walls = np.dstack([np.where(dbg["ink"], 0, 255).astype(np.uint8)] * 3)
        walls[dbg["bridges"]] = (0, 0, 255)  # bridged wall gaps in red
        cv2.imwrite(str(out / "bridges.png"), walls)

    # share of colored-fill pixels that ended up in a region: with no labels, the best sign of dropped buildings
    coverage = float((dbg["fill"] & (labels > 0)).sum() / max(dbg["fill"].sum(), 1))
    stats = {"sheet": str(a.sheet), "crop": [x, y, w, h], "params": vars(p), "city_knobs": city,
             "paper_rgb": [round(float(c), 1) for c in paper], "regions": n,
             "fill_coverage": round(coverage, 4), "seconds": round(seconds, 2),
             "flagged": {f: sum(v["flag"] == f for v in flags.values()) for f in ("round", "dark", "pale")},
             "suspect": sum(v["suspect"] for v in flags.values())}
    if a.gt and Path(a.gt).is_dir():
        # labeled sheet: score only inside its extents, away from ignore polygons
        ignore = load_config()["labels"]["ignore_value"]
        inst = cv2.imread(str(Path(a.gt) / "instances.png"), cv2.IMREAD_UNCHANGED)[y:y + h, x:x + w]
        valid = cv2.imread(str(Path(a.gt) / "valid.png"), cv2.IMREAD_UNCHANGED)[y:y + h, x:x + w] > 0
        valid &= inst != ignore
        gt = np.where(valid, inst, 0).astype(np.int32)
        inside = np.bincount(labels.ravel(), weights=valid.ravel(), minlength=n + 1) > 0.5 * np.maximum(area, 1)
        inside[0] = False
        scored = np.where(inside[labels], labels, 0)
        stats["checks"] = overseg_checks(np.unique(gt, return_inverse=True)[1].reshape(gt.shape),
                                         np.unique(scored, return_inverse=True)[1].reshape(scored.shape))
    elif a.gt:
        gt = rasterize(load_polygons(a.gt), (H, W))[y:y + h, x:x + w]
        stats["checks"] = overseg_checks(gt, labels)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"{n} regions, fill coverage {coverage:.1%}, {seconds:.1f} s, paper RGB {stats['paper_rgb']}, "
          f"flagged {stats['flagged']}, suspect {stats['suspect']}")
    for k, v in stats.get("checks", {}).items():
        print(f"  {k:28s} {v}")
    print(f"-> {out.relative_to(ROOT)}")


def number(s):
    f = float(s)
    return int(f) if f.is_integer() else f


if __name__ == "__main__":
    main()
