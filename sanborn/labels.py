"""Load and check label polygons, rasterize instance maps, QA overlays, dissolve merged regions by group (guide §4, §9.5).

examples:
  uv run dissolve sanborn01500_003_0012
  (same as: uv run python -m sanborn.labels dissolve sanborn01500_003_0012)

A sheet's labels live in data/labels/<sheet>/, written by the labeling tool (sanborn.labeltool):
  regions.png       the over-segmenter's region map (uint16), never changed; cuts are replayed on it
  regions.geojson   the over-segmenter's regions with their flag / suspect properties (read-only copy)
  edits.json        groups {region: group}, drop, reviewed, extents, ignore polygons, active_seconds,
                    walls [{line, region, flip}] (drawn walls, applied after the fit: `apply_wall`),
                    rev (save version: the tool refuses saves from a tab built on an older one)
  cuts.json         drawn cuts in order, each with its pieces {new region: parent region}

`dissolve` turns them into training labels (§4.4): instances.png (uint16, 0 = background, the
config's ignore_value = ignore), valid.png (255 = labeled area), buildings.geojson, summary.json
and qa.jpg, fitted.png / fitted.json (the buildings before the drawn walls and their regions, for the
labeling tool's final view). summary.json holds n_buildings / n_party_walls / split (the shared manifest is not
written, so two labelers pushing different sheets never conflict). On the way it moves the
outlines onto the drawn walls (`fit_to_walls`, config labels.fit; `--no-fit` skips it).
"""
import argparse
import json

import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union

from sanborn import ROOT, load_config, manifest

LABELS = ROOT / "data" / "labels"


def load_polygons(path):
    """Shapely geometries from a GeoJSON in pixel coordinates."""
    with open(path) as f:
        return [shape(feat["geometry"]) for feat in json.load(f)["features"]]


def rasterize(geoms, size):
    """Instance map (int32) of shape `size`: building i+1 at its pixels, 0 elsewhere, holes stay 0."""
    out = np.zeros(size, np.int32)
    for i, g in enumerate(geoms, 1):
        for poly in getattr(g, "geoms", [g]):
            cv2.fillPoly(out, [np.round(poly.exterior.coords).astype(np.int32)], i)
            for hole in poly.interiors:
                cv2.fillPoly(out, [np.round(hole.coords).astype(np.int32)], 0)
    return out


def polygonize(labels, simplify=1.0, offset=(0, 0)):
    """{id: geometry} for every label > 0, outer rings plus holes, simplified with Douglas-Peucker.

    `offset` (x, y) is added to all coordinates, so a crop's polygons land in sheet coordinates.
    """
    out = {}
    for i, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None:
            continue
        mask = (labels[sl] == i).astype(np.uint8)
        contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        dx, dy = sl[1].start + offset[0], sl[0].start + offset[1]
        ring = lambda c: c[:, 0, :] + (dx, dy)
        polys = []
        for j, c in enumerate(contours):
            if hier[0][j][3] != -1 or len(c) < 3:  # holes are handled with their parent
                continue
            holes, k = [], hier[0][j][2]
            while k != -1:
                if len(contours[k]) >= 3:
                    holes.append(ring(contours[k]))
                k = hier[0][k][0]
            polys.append(Polygon(ring(c), holes))
        if not polys:
            continue
        g = polys[0] if len(polys) == 1 else unary_union(polys)
        g = g.simplify(simplify, preserve_topology=True)
        out[i] = g if g.is_valid else g.buffer(0)
    return out


def write_geojson(path, geoms, props=None, header=None):
    """One feature per {id: geometry}; `props` maps id to extra properties."""
    feats = [{"type": "Feature", "properties": {"id": int(i), **(props or {}).get(i, {})},
              "geometry": mapping(g)} for i, g in geoms.items()]
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", **(header or {}), "features": feats}, f)


def qa_overlay(rgb, labels, seed=0):
    """Each region in its own random color over a faded copy of the sheet, with region outlines."""
    colors = np.random.default_rng(seed).integers(40, 230, (labels.max() + 1, 3)).astype(np.uint16)
    out = (np.asarray(rgb, np.uint16) * 2 + 3 * 255) // 5  # fade toward white
    m = labels > 0
    out[m] = (out[m] * 2 + colors[labels[m]] * 3) // 5
    edge = np.zeros(labels.shape, bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:] |= labels[1:] != labels[:-1]
    out[edge] = 30
    return out.astype(np.uint8)


# --- cuts: splitting a region the over-segmenter left merged (§9.5 step 4) ---

def cut(regions, line, next_id, min_piece=50):
    """Split every region the polyline `line` [(x, y), ...] runs fully across, in place.

    Each split region is replaced by new ids from `next_id` on. Pixels under the line and pieces
    under `min_piece` px join the nearest remaining piece, so no pixel changes building. Returns
    {new id: parent id}; empty if the line separated nothing. Raises ValueError, changing nothing,
    if the new ids don't fit in the map's dtype.
    """
    H, W = regions.shape
    pts = np.round(np.asarray(line, float)).astype(np.int32).reshape(-1, 2)
    pts = np.clip(pts, 0, (W - 1, H - 1))  # points off the sheet land on its border
    if len(pts) < 2:
        return {}
    x0, y0 = np.maximum(pts.min(0) - 2, 0)
    x1, y1 = np.minimum(pts.max(0) + 3, (W, H))
    stroke = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.polylines(stroke, [pts - (x0, y0)], False, 1, thickness=2)
    touched = np.unique(regions[y0:y1, x0:x1][stroke > 0])

    splits = []  # work out every split first, so a failure leaves the map untouched
    for r in touched[touched > 0]:
        ys, xs = np.nonzero(regions == r)
        sl = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        own = regions[sl] == r
        line_mask = np.zeros(own.shape, np.uint8)
        cv2.polylines(line_mask, [pts - (xs.min(), ys.min())], False, 1, thickness=2)
        parts, _ = ndimage.label(own & (line_mask == 0))  # 4-connected: a 2 px line separates
        size = np.bincount(parts.ravel())
        size[0] = 0
        big = np.flatnonzero(size >= min_piece)
        if len(big) < 2:
            continue
        seeds = np.where(np.isin(parts, big), parts, 0)
        _, (iy, ix) = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
        splits.append((int(r), sl, own, seeds[iy, ix], big))
    if next_id + sum(len(s[4]) for s in splits) - 1 > np.iinfo(regions.dtype).max:
        raise ValueError(f"region ids would pass {np.iinfo(regions.dtype).max}")

    pieces = {}
    for r, sl, own, nearest, big in splits:
        view = regions[sl]
        for b in big:
            view[own & (nearest == b)] = next_id
            pieces[next_id] = r
            next_id += 1
    return pieces


def uncut(regions, pieces):
    """Undo a cut in place: its pieces {new id: parent id} go back to the parent id."""
    lut = np.arange(max(int(regions.max()), *pieces) + 1, dtype=regions.dtype)
    lut[list(pieces)] = list(pieces.values())
    regions[:] = lut[regions]


def load_regions(d, min_piece):
    """The working region map of a label folder and its cuts: regions.png with cuts.json replayed.

    Replaying instead of saving the edited map keeps a cut to a few bytes on disk. Each cut is
    replayed with the min_piece it was drawn with and must give the same pieces again, otherwise the
    edits would point at wrong regions.
    """
    regions = cv2.imread(str(d / "regions.png"), cv2.IMREAD_UNCHANGED)
    cuts = json.loads((d / "cuts.json").read_text())["cuts"] if (d / "cuts.json").exists() else []
    for i, c in enumerate(cuts):
        pieces = cut(regions, c["line"], min(map(int, c["pieces"])), c.get("min_piece", min_piece))
        if {str(k): v for k, v in pieces.items()} != c["pieces"]:
            raise ValueError(f"cut {i} in {d / 'cuts.json'} no longer gives the same pieces")
    return regions, cuts


def neighbors(regions, reach, ids=None):
    """{region: [regions within `reach` px]} for all regions, or only `ids`.

    Walls between regions of one building are 1-8 px of ink, so these are the pairs a merge may join.
    """
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1,) * 2)
    H, W = regions.shape
    out = {}
    for i, sl in enumerate(ndimage.find_objects(regions), 1):
        if sl is None or (ids is not None and i not in ids):
            continue
        ys, xs = sl
        c = np.s_[max(ys.start - reach, 0):min(ys.stop + reach, H), max(xs.start - reach, 0):min(xs.stop + reach, W)]
        near = cv2.dilate((regions[c] == i).astype(np.uint8), disk) > 0
        found = np.unique(regions[c][near])
        out[i] = [int(j) for j in found if j and j != i]
    return out


# --- dissolve: regions + edits -> one instance per building (§9.5 step 5, §4.1, §4.4) ---

def building_map(regions, groups, drop):
    """Instance map (int32) from the region map, and {building: [region ids]}.

    Regions sharing a group become one building, dropped regions become background, and a region
    without a group is a building on its own (§9.5).
    """
    present = np.flatnonzero(np.bincount(regions.ravel()))
    lut = np.zeros(int(regions.max()) + 1, np.int32)
    index, members = {}, {}
    for r in present[present > 0]:
        r = int(r)
        if r in drop:
            continue
        b = index.setdefault(("group", groups[r]) if r in groups else ("region", r), len(index) + 1)
        lut[r] = b
        members.setdefault(b, []).append(r)
    return lut[regions], members


def close_cracks(inst, regions, members, gap):
    """Give a building the unclaimed ink between its own regions, in place.

    Neighboring regions are often separated by 1-8 px of wall ink that no region grew into.
    Left alone, a merged building would carry background strips along its interior walls, which
    would teach the network that those walls separate buildings. The building's closing fills
    gaps up to 2 x `gap` px; a filled patch is kept only if part of it lies within `gap` + 1 px of
    two of its regions, so notches in a single region's outline stay as drawn.
    """
    square = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * gap + 1,) * 2)  # a disk leaves notches at outer walls
    reach = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 3,) * 2)
    H, W = inst.shape
    objs = ndimage.find_objects(inst)
    for b, rs in members.items():
        if len(rs) < 2 or b > len(objs) or objs[b - 1] is None:
            continue
        ys, xs = objs[b - 1]
        sl = np.s_[max(ys.start - gap - 2, 0):min(ys.stop + gap + 2, H),
                   max(xs.start - gap - 2, 0):min(xs.stop + gap + 2, W)]
        own = (inst[sl] == b).astype(np.uint8)
        near = np.zeros(own.shape, np.uint8)
        for r in rs:
            near += cv2.dilate((regions[sl] == r).astype(np.uint8), reach) > 0
        closed = cv2.morphologyEx(own, cv2.MORPH_CLOSE, square) > 0
        view = inst[sl]
        patches, _ = ndimage.label(closed & (own == 0) & (view == 0))
        between = np.unique(patches[near >= 2])
        view[np.isin(patches, between[between > 0])] = b


def fill_holes(inst, max_hole):
    """Fill background pockets of up to `max_hole` px enclosed by one building, in place.

    Pockets holding another building are left alone, as are bigger ones (light courts). A dropped
    region enclosed by the building (a symbol printed inside it) is filled too: it isn't a building,
    but the area under it is.
    """
    for b, sl in enumerate(ndimage.find_objects(inst), 1):
        if sl is None:
            continue
        view = inst[sl]
        own = view == b
        pockets, n = ndimage.label(ndimage.binary_fill_holes(own) & ~own)
        if not n:
            continue
        size = np.bincount(pockets.ravel(), minlength=n + 1)
        other = np.bincount(pockets.ravel(), weights=((view != 0) & ~own).ravel().astype(float), minlength=n + 1)
        ok = (size <= max_hole) & (other == 0)
        ok[0] = False
        view[ok[pockets]] = b


def snap_gaps(inst, gap):
    """Close gaps up to `gap` px wide between two different buildings, in place (§4.1).

    Otherwise the contact class won't appear there. A background pixel is filled when, along its
    row, column or a diagonal, one building lies on one side and another on the other side with at
    most `gap` px between them; it joins the nearer one. Outer walls and wider alleys stay as drawn.
    """
    top = np.iinfo(np.int32).max
    hi = ndimage.maximum_filter(inst, size=2 * gap + 1)
    lo = ndimage.minimum_filter(np.where(inst > 0, inst, top), size=2 * gap + 1)
    ys, xs = np.nonzero((inst == 0) & (hi > 0) & (hi != lo))
    H, W = inst.shape

    def first(dy, dx):
        """Building id and distance in steps of the first building seen from each pixel."""
        lab, k = np.zeros(len(ys), inst.dtype), np.full(len(ys), gap + 1)
        for step in range(1, gap + 1):
            y, x = ys + step * dy, xs + step * dx
            ok = (y >= 0) & (y < H) & (x >= 0) & (x < W)
            v = np.where(ok, inst[y.clip(0, H - 1), x.clip(0, W - 1)], 0)
            new = (lab == 0) & (v > 0)
            lab[new], k[new] = v[new], step
        return lab, k

    best, dist = np.zeros(len(ys), inst.dtype), np.full(len(ys), gap + 1)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        (la, ka), (lb, kb) = first(dy, dx), first(-dy, -dx)
        take = (la > 0) & (lb > 0) & (la != lb) & (ka + kb <= gap + 1)
        near, d = np.where(ka <= kb, la, lb), np.minimum(ka, kb)
        take &= d < dist
        best[take], dist[take] = near[take], d[take]
    inst[ys[best > 0], xs[best > 0]] = best[best > 0]


def party_walls(inst, min_len, ignore=None):
    """Number of building pairs sharing at least `min_len` px of boundary (contact class examples)."""
    keys = []
    for a, b in ((inst[:, 1:], inst[:, :-1]), (inst[1:], inst[:-1])):
        m = (a > 0) & (b > 0) & (a != b)
        if ignore is not None:
            m &= (a != ignore) & (b != ignore)
        lo, hi = np.minimum(a[m], b[m]).astype(np.int64), np.maximum(a[m], b[m]).astype(np.int64)
        keys.append(lo * 65536 + hi)
    _, count = np.unique(np.concatenate(keys), return_counts=True)
    return int((count >= min_len).sum())


def polygon_mask(polys, shape):
    """Boolean mask of a list of polygons [[(x, y), ...], ...] (no holes)."""
    mask = np.zeros(shape, np.uint8)
    for p in polys:
        if len(p) >= 3:
            cv2.fillPoly(mask, [np.round(np.asarray(p, float)).astype(np.int32)], 1)
    return mask > 0


# Wall fit (dissolve time): move the outlines onto the drawn walls. The over-segmenter's regions follow
# the colored fill, so outlines wobble, spill ~3 px past dashed sides and are notched wherever text or a
# symbol touches a wall (outputs/wall_fix_study, docs/boundary_study). Regions and edits are untouched.

def wall_masks(rgb):
    """The over-segmenter's ink, ink_all and fill masks and L* (uint8) of a full sheet, for `fit_to_walls`.
    About 3 s and 1.5 GB on a full sheet."""
    from sanborn.oversegment import ink_masks, params  # oversegment imports this module
    m = ink_masks(np.asarray(rgb), params())
    return {"ink": m["ink"], "ink_all": m["ink_all"], "fill": m["fill"],
            "L": np.clip(m["L"], 0, 100).astype(np.uint8)}


def dash_lines(ink_all, ink, gap=15, min_len=45, max_thick=7, min_dashes=2):
    """Dashed lines (porch and shed sides) completed into lines: rows or columns of at least `min_dashes`
    thin elongated blobs only `ink_all` holds, with gaps up to `gap` px, at least `min_len` px long
    (counting solid ink the line runs into). Returns the new pixels: dashes and the gaps between them.
    A row of text is rejected by its thickness."""
    n, cc, st, _ = cv2.connectedComponentsWithStats((ink_all & ~ink).view(np.uint8), connectivity=8)
    w, h = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]
    out = np.zeros_like(ink)
    for horiz in (True, False):
        length, thick = (w, h) if horiz else (h, w)
        dash = (length >= 5) & (thick <= max_thick) & (length >= 1.8 * thick)
        dash[0] = False
        d = dash[cc]
        line = (1, gap) if horiz else (gap, 1)
        closed = cv2.morphologyEx((d | ink).view(np.uint8), cv2.MORPH_CLOSE, np.ones(line, np.uint8))
        line = (1, min_len) if horiz else (min_len, 1)
        o = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones(line, np.uint8)).view(bool) & ~ink
        both = d & o
        if not both.any():
            continue
        n2, cc2, st2, _ = cv2.connectedComponentsWithStats(o.view(np.uint8), connectivity=8)
        pairs = np.unique(np.stack([cc2[both], cc[both]], 1), axis=0)
        good = np.bincount(pairs[:, 0], minlength=n2) >= min_dashes
        good &= st2[:, cv2.CC_STAT_HEIGHT if horiz else cv2.CC_STAT_WIDTH] <= max_thick
        good[0] = False
        out |= good[cc2]
    return out


def wall_lines(masks, dashes, length):
    """(straight, walls): horizontal and vertical pieces of ink plus completed dashes at least `length` px
    long, with the dashes themselves; `walls` adds the over-segmenter's bridged wall gaps. Text and symbol
    strokes are not walls, so they can't shape an outline."""
    from sanborn.oversegment import bridge_walls, params, straight_pieces
    p = params()
    h, v = straight_pieces(masks["ink"] | dashes, length)
    straight = h | v | dashes
    return straight, straight | bridge_walls(masks["ink"], p.bridge_length, p.bridge_gap, p.bridge_span)


def complete_cells(inst, masks, walls, cover=0.6, other=0.05, min_width=8, max_piece=800):
    """Give each building the pieces of the wall cell it mostly covers that text or a symbol touching a
    wall cut off (the wedge behind a "1" or "2"). Returns a new map; only background pixels change.

    Cells are the areas enclosed by straight wall lines alone. A building takes the colored fill, text ink
    and wall halo of a cell when it covers at least `cover` of the cell and other buildings at most `other`.
    Only enclosed pieces of up to `max_piece` px are taken (a bigger or open piece is a dropped neighbor,
    such as a gallery strip, behind a line that wasn't found), and cells narrower than `min_width` px (the
    paper strip of a double wall) never are.
    """
    out = inst.copy()
    W = cv2.morphologyEx(walls.view(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))  # corner gaps
    W = cv2.dilate(W, np.ones((3, 3), np.uint8)).view(bool)
    n, cells = cv2.connectedComponents((~W).view(np.uint8), connectivity=4)
    dt = cv2.distanceTransform((cells > 0).view(np.uint8), cv2.DIST_L2, 3)
    wide = np.zeros(n, bool)
    wide[np.unique(cells[dt >= min_width / 2])] = True
    B = out > 0
    area = np.bincount(cells.ravel(), minlength=n)
    pair, k = np.unique(cells[B].astype(np.int64) * (int(out.max()) + 1) + out[B], return_counts=True)
    c, b = pair // (int(out.max()) + 1), pair % (int(out.max()) + 1)
    covered = np.bincount(c, weights=k, minlength=n)
    top = np.zeros(n, np.int64)
    lut = np.zeros(n, np.int32)
    for ci, bi, ki in zip(c, b, k):
        if ki > top[ci]:
            top[ci], lut[ci] = ki, bi
    ok = wide & (top >= cover * area) & (covered - top <= other * area)
    ok[0] = False
    lut[~ok] = 0
    halo = cv2.dilate(W.view(np.uint8), np.ones((7, 7), np.uint8)).view(bool)  # light band along the walls
    cand = (out == 0) & (lut[cells] > 0) & (masks["fill"] | masks["ink_all"] | halo)

    pn, piece, pst, _ = cv2.connectedComponentsWithStats(cand.view(np.uint8), connectivity=8)
    k3 = np.ones((3, 3), np.uint8)
    grown = cv2.dilate(piece.astype(np.float32), k3).astype(np.int32)  # piece id next to each pixel
    border = ~cand & (grown > 0)
    open_px = np.bincount(grown[border & ~W & (out == 0)], minlength=pn)
    total = np.bincount(grown[border], minlength=pn)
    bad = (pst[:, cv2.CC_STAT_AREA] > max_piece) | (open_px > np.maximum(0.1 * total, 3))
    bad[0] = False
    cand &= ~bad[piece]
    out[cand] = lut[cells][cand]
    if cand.any():
        # the 1-3 px between the new pixels and the wall ink lie inside W: grow across them, up to the
        # ink, so the claim step can reach the wall
        allowed = (out == 0) & W & ~masks["ink_all"]
        got = _grow_within(cand, allowed | cand, 3, k3) & allowed
        _, (iy, ix) = ndimage.distance_transform_edt(~cand, return_indices=True)
        out[got] = out[iy[got], ix[got]]
    return out


def trim_beyond_walls(inst, straight, radius=4):
    """Drop thin pieces (inscribed radius <= `radius` px) of a building that lie beyond one of its straight
    walls or completed dashed lines on open background: fill spilled past a dashed porch side. A piece
    next to another building (the paper strip of a double wall) is kept."""
    out = inst.copy()
    bg = (inst == 0) & ~straight
    H, W = inst.shape
    k3, k5 = np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8)
    for b, sl in enumerate(ndimage.find_objects(inst), 1):
        if sl is None:
            continue
        ys, xs = sl
        s = np.s_[max(ys.start - 2, 0):min(ys.stop + 2, H), max(xs.start - 2, 0):min(xs.stop + 2, W)]
        v = inst[s]
        own = (v == b) & ~straight[s]
        lab, n = ndimage.label(own)
        if n < 2:
            continue
        size = np.bincount(lab.ravel())
        size[0] = 0
        r = ndimage.maximum(ndimage.distance_transform_edt(own), lab, np.arange(n + 1))
        at_bg = np.bincount(lab[own & (cv2.dilate(bg[s].view(np.uint8), k3) > 0)], minlength=n + 1) > 0
        other = ((v != b) & (v != 0)).view(np.uint8)
        at_other = np.bincount(lab[own & (cv2.dilate(other, k5) > 0)], minlength=n + 1) > 0
        thin = (r <= radius) & at_bg & ~at_other
        thin[0] = thin[size.argmax()] = False
        out[s][thin[lab]] = 0
    return out


def _grow_within(seed, allowed, steps, k):
    got = seed & allowed
    for _ in range(steps):
        nxt = (cv2.dilate(got.view(np.uint8), k) > 0) & allowed
        if (nxt == got).all():
            break
        got = nxt
    return got


def _claim(inst, ink, reach, L, halo_L):
    """Background ink touching a building, within `reach` px through ink, to the nearest building. The
    light outer halo (L* >= halo_L) reached from outside stays background, so the outline sits on the
    wall's half-darkness outer edge."""
    allowed = ink & (inst == 0)
    k8 = np.ones((3, 3), np.uint8)
    got = _grow_within(cv2.dilate((inst > 0).view(np.uint8), k8) > 0, allowed, reach - 1, k8)
    d, (iy, ix) = ndimage.distance_transform_edt(inst == 0, return_indices=True)
    got &= d <= reach
    halo = got & (L >= halo_L)
    k4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    outside = cv2.dilate(((inst == 0) & ~got).view(np.uint8), k4) > 0
    got &= ~_grow_within(outside, halo, reach, k4)
    out = inst.copy()
    out[got] = inst[iy[got], ix[got]]
    return out


def _peel(inst, ink_all, fill, depth):
    """Building pixels within `depth` px of background that are plain paper, reached from outside: the
    over-segmenter's grow past dashed or open sides."""
    bg = (inst == 0).view(np.uint8)
    d_in = cv2.distanceTransform(1 - bg, cv2.DIST_L2, 3)
    paper = (inst > 0) & ~fill & ~ink_all & (d_in <= depth)
    k4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    out = inst.copy()
    out[_grow_within(cv2.dilate(bg, k4) > 0, paper, depth, k4)] = 0
    return out


def _smooth(inst, open_size, close_size, protect):
    """Per building, a square opening (spurs from text or lot lines; never within `protect` px of another
    building, which counts as solid) and a square closing (gaps between dashes; adds background only)."""
    out = inst.copy()
    H, W = inst.shape
    pad = max(open_size, close_size, 2 * protect + 1) + 2
    ko, kc = np.ones((open_size,) * 2, np.uint8), np.ones((close_size,) * 2, np.uint8)
    kp = np.ones((2 * protect + 1,) * 2, np.uint8)
    for b, sl in enumerate(ndimage.find_objects(inst), 1):
        if sl is None:
            continue
        ys, xs = sl
        v = out[max(ys.start - pad, 0):min(ys.stop + pad, H), max(xs.start - pad, 0):min(xs.stop + pad, W)]
        own = (v == b).astype(np.uint8)
        other = ((v != b) & (v != 0)).astype(np.uint8)
        if open_size > 1:
            keep = cv2.morphologyEx(own | other, cv2.MORPH_OPEN, ko) > 0
            if protect:
                keep |= cv2.dilate(other, kp) > 0
            lost = (own > 0) & ~keep
            if lost.any():
                n0 = ndimage.label(own)[1]
                v[lost] = 0
                lab, n = ndimage.label(v == b)
                if n > n0:  # the opening cut a piece off: drop the new crumbs (< 50 px), smallest first
                    sz = np.bincount(lab.ravel())
                    sz[0] = 0
                    for j in np.argsort(sz)[: n - n0 + 1]:
                        if j and sz[j] < 50 and n > n0:
                            v[lab == j] = 0
                            n -= 1
            own = (v == b).astype(np.uint8)
        if close_size > 1:
            v[(cv2.morphologyEx(own, cv2.MORPH_CLOSE, kc) > 0) & (v == 0)] = b
    return out


def _fill_shared_pockets(inst, max_px):
    """Background pockets of up to `max_px` px enclosed only by buildings, each pixel to its nearest
    building, so the contact band along a shared wall is not broken. In place."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((inst == 0).view(np.uint8), connectivity=4)
    H, W = inst.shape
    k4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for i in np.flatnonzero(st[1:, cv2.CC_STAT_AREA] <= max_px) + 1:
        x, y, w, h = st[i, :4]
        if x == 0 or y == 0 or x + w == W or y + h == H:
            continue
        s = np.s_[y - 1:y + h + 1, x - 1:x + w + 1]
        m = lab[s] == i
        owners = inst[s][(cv2.dilate(m.view(np.uint8), k4) > 0) & ~m]
        if (owners == 0).any() or len(np.unique(owners)) < 2:
            continue
        v = inst[s]
        _, (iy, ix) = ndimage.distance_transform_edt(v == 0, return_indices=True)
        v[m] = v[iy[m], ix[m]]


def fit_to_walls(inst, masks, reach=6, peel=5, open_size=5, close_size=9, protect=3, halo_L=68,
                 shared_pocket=120, max_hole=3000, dash_gap=15, dash_len=45, dash_thick=7, wall_len=21,
                 cell_cover=0.6, cell_other=0.05, cell_width=8, cell_piece=800, trim_radius=4):
    """Outlines of a dissolved instance map moved onto the drawn walls; `masks` from `wall_masks`.

    In order: dashed lines are completed into walls; cut-off wedges are given back (`complete_cells`);
    spill beyond straight and dashed walls (`trim_beyond_walls`) and plain paper in the outer `peel` px are
    removed; wall ink within `reach` px is claimed, nearest building first, so shared walls split down the
    middle; outlines are smoothed; pockets are filled. Buildings only gain background or give pixels back
    to it: no pixel changes from one building to another. About 30 s and 3.5 GB on a full sheet.
    """
    inst = inst.astype(np.int32)
    dashes = dash_lines(masks["ink_all"], masks["ink"], dash_gap, dash_len, dash_thick)
    straight, walls = wall_lines(masks, dashes, wall_len)
    out = complete_cells(inst, masks, walls, cell_cover, cell_other, cell_width, cell_piece)
    out = trim_beyond_walls(out, straight, trim_radius)
    del straight, walls
    ink, ink_all = masks["ink"] | dashes, masks["ink_all"] | dashes
    if peel:
        out = _peel(out, ink_all, masks["fill"], peel)
    out = _claim(out, ink, reach, masks["L"], halo_L)
    out = _claim(out, ink_all & ~ink, max(reach - 2, 2), masks["L"], halo_L)
    out = _smooth(out, open_size, close_size, protect)
    fill_holes(out, max_hole)
    if shared_pocket:
        _fill_shared_pockets(out, shared_pocket)
    moved = (inst > 0) & (out > 0) & (out != inst)
    out[moved] = inst[moved]
    return out


# --- drawn walls (dissolve time, after the fit): the labeler's fixes for outlines the fit got wrong ---

def apply_wall(inst, b, line, flip=False, width=3, snap=8):
    """Move building `b`'s outline onto a wall drawn as the polyline `line` [(x, y), ...], in place.

    The line follows the outer edge of the real wall. An end within `snap` px of the building is joined
    to it. The line itself and the background it closes off against the building (a missing corner, an
    edge that stops short) join `b`. Where the line runs across the building, the largest part stays
    (`flip`: the second largest) and the others become background (spill past the wall). Pixels of other
    buildings never change. Returns the number of pixels changed (0: the line did nothing).
    """
    H, W = inst.shape
    pts = np.clip(np.round(np.asarray(line, float)).astype(np.int64).reshape(-1, 2), 0, (W - 1, H - 1))
    ys, xs = np.nonzero(inst == b)
    if len(pts) < 2 or not len(ys):
        return 0
    pad = snap + width + 3
    x0, y0 = max(min(xs.min(), pts[:, 0].min()) - pad, 0), max(min(ys.min(), pts[:, 1].min()) - pad, 0)
    x1, y1 = min(max(xs.max(), pts[:, 0].max()) + pad + 1, W), min(max(ys.max(), pts[:, 1].max()) + pad + 1, H)
    view = inst[y0:y1, x0:x1]
    own, free = view == b, view == 0
    pts = pts - (x0, y0)

    dist, (iy, ix) = ndimage.distance_transform_edt(~own, return_indices=True)
    ends = []
    for x, y in (pts[0], pts[-1]):
        ends.append([(ix[y, x], iy[y, x])] if 0 < dist[y, x] <= snap else [])
    pts = np.array(ends[0] + pts.tolist() + ends[1], np.int32)
    stroke = np.zeros(own.shape, np.uint8)
    cv2.polylines(stroke, [pts], False, 1, thickness=max(1, (width + 1) // 2))  # cv2 draws 2t - 1 px wide
    stroke = stroke > 0
    wall = stroke & (own | free)

    # background pockets the wall closes off against the building (not the building's own courtyards)
    closed = own | wall
    pockets, n = ndimage.label(ndimage.binary_fill_holes(closed) & ~closed & free)
    near_line = ndimage.binary_dilation(stroke)
    keep_pocket = np.zeros(n + 1, bool)
    keep_pocket[np.unique(pockets[near_line & (pockets > 0)])] = True
    keep_pocket[0] = False
    grown = own | keep_pocket[pockets]

    # parts of the building on either side of the line: the largest stays
    parts, n = ndimage.label(grown & ~stroke)
    size = np.bincount(parts.ravel(), minlength=n + 1)
    touching = np.unique(parts[near_line & (parts > 0)])
    order = touching[np.argsort(-size[touching], kind="stable")]
    if flip and len(order) < 2:
        return 0
    keep = np.ones(n + 1, bool)
    keep[order] = False
    if len(order):
        keep[order[1 if flip else 0]] = True
    keep[0] = False
    body = keep[parts]
    # the half of the line facing the kept body joins it, so the outline lands on the line; the far
    # half and tails past the building's ends stay out
    near = wall & (ndimage.distance_transform_cdt(~body, "chessboard") <= width // 2 + 1)  # square corners
    # a line running past the building's corner would leave a nub there: keep only the line that runs along it
    along = cv2.morphologyEx((body | near).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    new = body | (near & along)
    lost, gained = own & ~new, new & ~own
    view[lost] = 0
    view[gained] = b
    return int(lost.sum() + gained.sum())


def apply_walls(inst, walls, members, width=3, snap=8, only=None):
    """Apply the drawn walls in order, in place; each wall names a region, and changes that region's building.

    `members` {building: [regions]} is `inst`'s. `only` limits it to those buildings. Returns per wall the
    pixels it changed, or None when its region is in no building (dropped, cut since) or not in `only`.
    """
    owner = {r: b for b, rs in members.items() for r in rs}
    out = []
    for w in walls:
        b = owner.get(int(w["region"]))
        if b is None or (only is not None and b not in only):
            out.append(None)
            continue
        out.append(apply_wall(inst, b, w["line"], w.get("flip", False), width, snap))
    return out


def dissolve(regions, edits, cfg, masks=None, keep_base=False):
    """Training labels from a region map and the labeler's edits.

    Returns (instances uint16, valid bool, info). Without extents the whole sheet is labeled;
    with extents only buildings mostly inside them are kept, and the area of buildings mostly
    outside is marked not labeled, so no building is half labeled. With `masks` (from `wall_masks`)
    and a `fit` block in `cfg`, outlines are moved onto the drawn walls first (`fit_to_walls`). The walls
    the labeler drew (edits "walls") come after the fit, so it can't undo them. `keep_base` adds the
    buildings before those walls to info (base, base_members): the labeler's final view starts from them.
    """
    groups = {int(k): v for k, v in edits.get("groups", {}).items()}
    drop = set(map(int, edits.get("drop", [])))
    inst, members = building_map(regions, groups, drop)
    close_cracks(inst, regions, members, cfg["crack_gap"])
    fill_holes(inst, cfg["max_hole"])
    fitted = masks is not None and bool(cfg.get("fit"))
    if fitted:
        inst = fit_to_walls(inst, masks, max_hole=cfg["max_hole"], **cfg["fit"])

    info = {"extents": len(edits.get("extents", [])), "crossing_extent": [], "removed_at_edge": 0,
            "wall_fit": fitted}
    if keep_base:
        info.update(base=inst.copy(), base_members={b: list(rs) for b, rs in members.items()})
    walls = edits.get("walls", [])
    done = apply_walls(inst, walls, members, cfg["wall_width"], cfg["wall_snap"])
    info.update(walls=sum(d is not None for d in done),
                walls_skipped=[i for i, d in enumerate(done) if d is None],
                walls_noop=[i for i, d in enumerate(done) if d == 0])
    if edits.get("extents"):
        inside = polygon_mask(edits["extents"], inst.shape)
        n = int(inst.max()) + 1
        area = np.bincount(inst.ravel(), minlength=n)
        share = np.bincount(inst[inside], minlength=n) / np.maximum(area, 1)
        keep = share >= cfg["extent_share"]
        keep[0] = False
        crossing = (share > 0.02) & (share < 0.98)
        crossing[0] = False
        info["removed_at_edge"] = int((crossing & ~keep).sum())
        info["crossing_extent"] = np.flatnonzero(crossing & keep).tolist()
        valid = (inside & (inst == 0)) | keep[inst]
        inst[~keep[inst]] = 0
    else:
        valid = np.ones(inst.shape, bool)
    snap_gaps(inst, cfg["snap_gap"])
    valid |= inst > 0
    ignored = polygon_mask(edits.get("ignore", []), inst.shape)
    inst[ignored] = 0  # counted as neither building nor background, so buildings and walls are counted without them

    # number buildings 1..N and keep track of their regions
    ids = np.flatnonzero(np.bincount(inst.ravel()))
    ids = ids[ids > 0]
    lut = np.zeros(int(inst.max()) + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    crossing = [int(lut[b]) for b in info["crossing_extent"] if b < len(lut) and lut[b]]
    members = {int(lut[b]): rs for b, rs in members.items() if b < len(lut) and lut[b]}
    inst = lut[inst]
    if len(ids) >= cfg["ignore_value"]:
        raise ValueError(f"{len(ids)} buildings do not fit below ignore_value {cfg['ignore_value']}")

    info.update(
        buildings=len(ids),
        party_walls=party_walls(inst, cfg["party_wall_min"]),
        regions=sum(len(rs) for rs in members.values()),
        merges=sum(len(rs) - 1 for rs in members.values()),
        dropped=len(drop),
        members=members,
        crossing_extent=crossing,
    )
    out = inst.astype(np.uint16)
    out[ignored] = cfg["ignore_value"]
    return out, valid, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("dissolve", help="turn a sheet's regions + edits into training labels")
    s.add_argument("sheet", help="sheet id, e.g. sanborn07492_002_0003 (a path to the .jp2 works too)")
    s.add_argument("--no-fit", action="store_true",
                   help="keep the region outlines as the over-segmenter drew them (skip the labels.fit wall fit)")
    a = ap.parse_args()

    from pathlib import Path
    from sanborn.data import load_sheet

    sheet = Path(a.sheet).stem
    d = LABELS / sheet
    if not (d / "edits.json").exists():
        raise SystemExit(f"no edits in {d.relative_to(ROOT)}; label the sheet first: uv run label {sheet}")
    cfg = load_config()["labels"]
    edits = json.loads((d / "edits.json").read_text())
    row = next((r for r in manifest.read() if r["sheet_id"] == sheet), {})
    split = "test" if row.get("city", "").lower() in cfg["test_cities"] else "train"
    if split == "train" and not edits.get("extents"):
        raise SystemExit(f"{sheet} has no extent yet: in the labeling tool, click 'Use whole sheet' in the panel "
                         "(step 1), then run dissolve again")
    regions, cuts = load_regions(d, cfg["min_piece"])
    jp2 = ROOT / "data" / "raw" / f"{sheet}.jp2"
    if not jp2.exists():
        raise SystemExit(f"The sheet image is missing: put {sheet}.jp2 into the data/raw folder (see README.md, step 3)")
    print(f"dissolving {sheet} (about a minute) ...")
    rgb = load_sheet(jp2)
    masks = None if a.no_fit or not cfg.get("fit") else wall_masks(rgb)

    inst, valid, info = dissolve(regions, edits, cfg, masks, keep_base=True)
    del masks
    # the buildings before the drawn walls, for the labeler's final view (O) and its wall tool (W)
    base = info.pop("base")
    if base.max() >= 2 ** 16:
        raise ValueError(f"{int(base.max())} buildings do not fit in fitted.png (uint16)")
    cv2.imwrite(str(d / "fitted.png"), base.astype(np.uint16))
    (d / "fitted.json").write_text(json.dumps({"members": {str(b): rs for b, rs in info.pop("base_members").items()},
                                               "edits_rev": edits.get("rev", 0)}))
    del base
    ign = cfg["ignore_value"]
    polys = polygonize(np.where(inst == ign, 0, inst), simplify=load_config()["polygons"]["simplify"])
    multi = [i for i, g in polys.items() if g.geom_type == "MultiPolygon"]
    props = {i: {"regions": info["members"].get(i, []), "area_px": int(g.area)} for i, g in polys.items()}

    cv2.imwrite(str(d / "instances.png"), inst)
    cv2.imwrite(str(d / "valid.png"), valid.astype(np.uint8) * 255)
    write_geojson(d / "buildings.geojson", polys, props, {"sheet": sheet})
    qa = qa_overlay(rgb, np.where(inst == ign, 0, inst))
    qa[~valid] = qa[~valid] // 3                        # not labeled: dark
    qa[inst == ign] = (qa[inst == ign] + (255, 220, 0)) // 2  # ignore: yellow
    cv2.imwrite(str(d / "qa.jpg"), cv2.cvtColor(qa, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])

    summary = {k: v for k, v in info.items() if k != "members"}
    summary.update(sheet=sheet, city=row.get("city", ""), split=split, cuts=len(cuts), ignore_polygons=len(edits.get("ignore", [])),
                   multi_part=multi, minutes=round(edits.get("active_seconds", 0) / 60, 1))
    (d / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"{info['buildings']} buildings, {info['party_walls']} party walls ({split}), "
          f"{info['merges']} merges, {info['dropped']} dropped, {len(cuts)} cuts, {summary['minutes']} min")
    if info["removed_at_edge"]:
        print(f"  {info['removed_at_edge']} buildings mostly outside an extent were left out (their area is not labeled)")
    if crossing := info["crossing_extent"]:
        print(f"  {len(crossing)} buildings cross an extent edge (ids {crossing[:10]}): move the edge onto the street")
    if multi:
        print(f"  {len(multi)} buildings are in several pieces (ids {multi[:10]}): check their groups")
    if info["walls"] or info["walls_skipped"]:
        print(f"  {info['walls']} drawn walls applied")
    if skipped := info["walls_skipped"]:
        print(f"  {len(skipped)} drawn walls skipped (walls {skipped[:10]}): their region was dropped or cut since; "
              "redraw them in the labeling tool")
    if noop := info["walls_noop"]:
        print(f"  {len(noop)} drawn walls changed nothing (walls {noop[:10]}): the merges under them changed; check them")
    print(f"-> {d.relative_to(ROOT)}  (open qa.jpg there to check the result)")


def dissolve_main():
    """`uv run dissolve <sheet>`: the same as `python -m sanborn.labels dissolve <sheet>`."""
    import sys
    sys.argv.insert(1, "dissolve")
    main()


if __name__ == "__main__":
    main()
