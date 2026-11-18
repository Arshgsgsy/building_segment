from pathlib import Path

import cv2
import numpy as np

from sanborn import load_config, synthetic
from sanborn.data import load_sheet
from sanborn.evaluate import overseg_checks
from sanborn.labels import load_polygons, polygonize, rasterize
from sanborn.oversegment import bridge_walls, drop_slivers, params, segment, straight_pieces


def test_synthetic_sheet_matches_table_14(tmp_path):
    synthetic.make(tmp_path)
    rgb = load_sheet(tmp_path / "synthetic_sheet.png")
    labels, _ = segment(rgb, params())
    gt = rasterize(load_polygons(tmp_path / "synthetic_gt.geojson"), rgb.shape[:2])

    c = overseg_checks(gt, labels)
    assert c["buildings"] == 11
    assert c["regions"] == 16
    assert c["under_segmented"] == 0
    assert c["buildings_with_region"] == 11
    assert c["regions_mostly_outside"] == 0
    assert c["regions_per_building_median"] == 1


def test_polygons_round_trip(tmp_path):
    synthetic.make(tmp_path)
    labels, _ = segment(load_sheet(tmp_path / "synthetic_sheet.png"), params())
    polys = polygonize(labels)
    assert len(polys) == labels.max()
    assert all(g.is_valid and not g.is_empty for g in polys.values())
    assert sum(len(g.interiors) for g in polys.values()) == 1  # the courtyard


def test_bridge_walls_closes_a_door_in_a_wall():
    ink = np.zeros((200, 400), bool)
    ink[100:103, 20:180] = ink[100:103, 220:380] = True   # wall with a 40 px opening
    ink[20:180, 20:23] = True                             # joined to a long wall: part of the network
    added = bridge_walls(ink, length=31, gap=60, span=150)
    assert added[100:103, 180:220].all()
    assert not added[:95].any() and not added[106:].any()


def test_drop_slivers_keeps_small_sheds():
    labels = np.zeros((200, 200), np.int32)
    labels[20:30, 20:180] = 1    # 10 px strip along a wall
    labels[60:85, 60:85] = 2     # 25 px shed
    out = drop_slivers(labels, r_max=9, circ_max=0.35)
    assert set(np.unique(out)) == {0, 1}
    assert (out[60:85, 60:85] == 1).all() and not out[20:30, 20:180].any()


def test_straight_pieces_stay_on_ink_for_even_lengths():
    ink = np.zeros((5, 100), bool)
    ink[2, 10:60] = True
    for length in (24, 25, 26):
        h, _ = straight_pieces(ink, length)
        assert (h == ink).all()


def test_hue_cut_splits_yellow_frame_from_green_neighbour():
    """Real San Francisco crop where yellow and green buildings meet at a dashed line: the hue cut must split them."""
    cfg = load_config()
    assert cfg["oversegment"]["hue_cut"] == 0 and cfg["oversegment"]["dash_cut"] == 0  # off for other cities
    sf = cfg["oversegment_by_city"]["san francisco"]
    assert sf["hue_cut"] > 0
    rgb = cv2.cvtColor(cv2.imread(str(Path(__file__).parent / "data" / "sf_restt_sal_lunch.png")), cv2.COLOR_BGR2RGB)
    paper = np.array([180.0, 178.0, 155.0])  # paper_color of the full sheet
    yellow, green = (100, 80), (100, 200)
    crop = (False,) * 4

    off, _ = segment(rgb, params(None, **{**sf, "hue_cut": 0}), paper, crop)
    assert off[yellow] > 0 and off[yellow] == off[green]

    on, _ = segment(rgb, params(None, **sf), paper, crop)
    assert on[yellow] > 0 and on[green] > 0 and on[yellow] != on[green]
    # each building stays whole: no cut inside one color (0: text holes)
    for window in (on[60:200, 20:120], on[60:200, 160:260]):
        assert len(np.unique(window[window > 0])) == 1
