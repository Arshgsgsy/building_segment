"""Build a synthetic Sanborn-like test sheet plus ground-truth GeoJSON (guide §9.3).

Run `uv run python -m sanborn.synthetic [out_dir]`.
"""
import json
import os
import sys

import cv2
import numpy as np

from sanborn import ROOT


def make(out):
    """Write synthetic_sheet.png and synthetic_gt.geojson (11 buildings) into `out`."""
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(1)
    H, W = 1800, 2400
    paper = np.array([228, 214, 172], np.float32)  # aged yellow paper (RGB)
    img = np.ones((H, W, 3), np.float32) * paper
    img += rng.normal(0, 4, img.shape)

    PINK = (214, 150, 160)
    YELLOW = (226, 205, 90)
    BLUE = (140, 170, 205)
    INK = (25, 22, 20)
    gt = []

    def rect(x0, y0, x1, y1, color, walls=3):
        cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), INK, walls)

    def poly(pts, color, walls=3):
        p = np.array(pts, np.int32)
        cv2.fillPoly(img, [p], color)
        cv2.polylines(img, [p], True, INK, walls)

    # Row of 5 attached buildings sharing party walls
    xs = [200, 380, 520, 700, 860, 1080]
    cols = [PINK, YELLOW, PINK, YELLOW, PINK]
    for i in range(5):
        rect(xs[i], 200, xs[i + 1], 520, cols[i])
        gt.append([[xs[i], 200], [xs[i + 1], 200], [xs[i + 1], 520], [xs[i], 520]])
    # faded party wall: punch a 2 px gap into the wall at x=700
    img[350:352, 697:704] = paper

    # Building with an attached den (den is part of the building in GT)
    poly([[200, 700], [600, 700], [600, 1000], [200, 1000]], PINK)
    poly([[600, 780], [720, 780], [720, 920], [600, 920]], YELLOW)  # den / wing
    gt.append([[200, 700], [600, 700], [600, 780], [720, 780], [720, 920],
               [600, 920], [600, 1000], [200, 1000]])

    # Detached dwelling with an interior porch line (thin line, same building)
    rect(900, 700, 1200, 1000, YELLOW)
    cv2.line(img, (900, 940), (1200, 940), INK, 2)
    gt.append([[900, 700], [1200, 700], [1200, 1000], [900, 1000]])

    # Small shed (~100 ft^2 at 6.25 px/ft is ~62 px wide)
    rect(1350, 760, 1412, 822, YELLOW)
    gt.append([[1350, 760], [1412, 760], [1412, 822], [1350, 822]])

    # Tiny outbuilding
    rect(1500, 900, 1530, 930, PINK)
    gt.append([[1500, 900], [1530, 900], [1530, 930], [1500, 930]])

    # Special building with X hatching
    rect(1600, 200, 2000, 520, BLUE)
    cv2.line(img, (1600, 200), (2000, 520), INK, 2)
    cv2.line(img, (2000, 200), (1600, 520), INK, 2)
    gt.append([[1600, 200], [2000, 200], [2000, 520], [1600, 520]])

    # Courtyard building (hole)
    rect(1500, 1150, 2100, 1650, PINK)
    rect(1700, 1300, 1900, 1500, tuple(int(c) for c in paper))
    gt_court = [[[1500, 1150], [2100, 1150], [2100, 1650], [1500, 1650]],
                [[1700, 1300], [1900, 1300], [1900, 1500], [1700, 1500]]]

    # Text inside buildings and on streets
    img = np.clip(img, 0, 255).astype(np.uint8)
    for (x, y, s) in [(240, 380, "2"), (420, 380, "D"), (560, 380, "S"), (980, 860, "D"),
                      (300, 860, "STORE"), (1150, 600, "MAIN ST."), (500, 1300, "126")]:
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, INK, 3)

    # Brown stain in the open (should not become a building)
    yy, xx = np.mgrid[0:H, 0:W]
    stain = np.exp(-(((xx - 600) / 220) ** 2 + ((yy - 1400) / 160) ** 2))
    img = img.astype(np.float32) * (1 - 0.35 * stain[..., None] * np.array([0.2, 0.45, 0.9]))

    # Lot lines (thin, close a yard against buildings)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.line(img, (200, 1000), (200, 1150), INK, 1)
    cv2.line(img, (200, 1150), (600, 1150), INK, 1)
    cv2.line(img, (600, 1000), (600, 1150), INK, 1)

    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    cv2.imwrite(os.path.join(out, "synthetic_sheet.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    feats = [{"type": "Feature", "properties": {"id": i + 1},
              "geometry": {"type": "Polygon", "coordinates": [p + [p[0]]]}}
             for i, p in enumerate(gt)]
    feats.append({"type": "Feature", "properties": {"id": len(feats) + 1},
                  "geometry": {"type": "Polygon",
                               "coordinates": [r + [r[0]] for r in gt_court]}})
    with open(os.path.join(out, "synthetic_gt.geojson"), "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    return len(feats)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "synthetic"
    print(make(out), "buildings")
