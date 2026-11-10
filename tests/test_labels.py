import json

import cv2
import numpy as np
import pytest

from sanborn import load_config
from sanborn.labels import apply_wall, apply_walls, cut, dissolve, fill_holes, load_regions, neighbors, uncut

CFG = load_config()["labels"]
IGN = CFG["ignore_value"]


@pytest.fixture(params=[False, True], ids=["plain", "wall_fit"])
def run_dissolve(request):
    """dissolve as is, and with the wall fit on masks of plain paper (no ink): the same assertions hold."""
    def run(regions, edits, cfg):
        if not request.param:
            return dissolve(regions, edits, cfg)
        H, W = regions.shape
        masks = {"ink": np.zeros((H, W), bool), "ink_all": np.zeros((H, W), bool), "fill": regions > 0,
                 "L": np.full((H, W), 97, np.uint8)}
        return dissolve(regions, edits, cfg, masks)
    return run


def two_regions(gap):
    """Two 60 x 76 px regions side by side, `gap` px of unclaimed ink between them."""
    regions = np.zeros((100, 220), np.uint16)
    regions[20:80, 20:96] = 1
    regions[20:80, 96 + gap:172 + gap] = 2
    return regions


def test_merging_fills_the_wall_between_regions(run_dissolve):
    inst, valid, info = run_dissolve(two_regions(8), {"groups": {"1": 7, "2": 7}}, CFG)
    assert info["buildings"] == 1 and info["merges"] == 1 and info["party_walls"] == 0
    assert (inst[20:80, 20:180] == 1).all()             # no background strip along the inner wall
    assert not inst[:20].any() and not inst[80:].any()  # outer edges unchanged
    assert valid.all()


def test_neighbors_are_snapped_and_counted_as_party_wall(run_dissolve):
    inst, _, info = run_dissolve(two_regions(2), {}, CFG)
    assert info["buildings"] == 2 and info["party_walls"] == 1
    assert (inst[20:80, 20:174] > 0).all()              # 2 px gap closed
    assert not inst[:20].any() and not inst[80:].any()  # without bumps past the fronts
    inst, _, info = run_dissolve(two_regions(8), {}, CFG)
    assert info["party_walls"] == 0 and not inst[20:80, 97:103].any()  # a wider gap stays


def test_drop_extent_and_ignore(run_dissolve):
    regions = two_regions(30)
    regions[85:95, 10:20] = 3
    edits = {"drop": [3], "extents": [[(0, 0), (110, 0), (110, 100), (0, 100)]],
             "ignore": [[(40, 40), (50, 40), (50, 50), (40, 50)]]}
    inst, valid, info = run_dissolve(regions, edits, CFG)
    assert info["buildings"] == 1 and info["dropped"] == 1
    assert not inst[85:95, 10:20].any() and valid[85:95, 10:20].all()  # dropped: labeled background
    assert not inst[20:80, 126:202].any() and not valid[20:80, 126:202].any()  # outside: not labeled
    assert (inst[41:50, 41:50] == IGN).all()


def test_fill_holes_keeps_light_courts():
    inst = np.zeros((300, 300), np.int32)
    inst[10:290, 10:290] = 1
    inst[30:40, 30:40] = 0      # 100 px pocket: filled
    inst[100:200, 100:200] = 0  # 10,000 px light court: kept
    fill_holes(inst, 3000)
    assert (inst[30:40, 30:40] == 1).all() and not inst[100:200, 100:200].any()


def test_cut_splits_and_uncut_restores():
    regions = np.zeros((100, 200), np.uint16)
    regions[20:80, 20:180] = 1
    before = regions.copy()
    pieces = cut(regions, [(100, 10), (100, 90)], next_id=2)
    assert sorted(pieces.items()) == [(2, 1), (3, 1)]
    assert set(np.unique(regions)) == {0, 2, 3}
    assert ((regions > 0) == (before > 0)).all()        # every pixel kept, line pixels included
    uncut(regions, pieces)
    assert (regions == before).all()
    assert cut(regions, [(100, 10), (100, 50)], next_id=2) == {}  # stops halfway: no split


def test_ignore_box_is_not_counted(run_dissolve):
    regions = two_regions(2)
    edits = {"ignore": [[(97, 10), (210, 10), (210, 90), (97, 90)]]}  # covers all of building 2
    inst, _, info = run_dissolve(regions, edits, CFG)
    assert info["buildings"] == 1 and info["party_walls"] == 0
    assert set(np.unique(inst)) == {0, 1, IGN}


def test_failed_cut_changes_nothing():
    regions = np.zeros((100, 200), np.uint16)
    regions[20:80, 20:180] = 1
    before = regions.copy()
    with pytest.raises(ValueError):
        cut(regions, [(100, 10), (100, 90)], next_id=65535)
    assert (regions == before).all()


def test_cuts_replay_from_disk(tmp_path):
    regions = np.zeros((100, 200), np.uint16)
    regions[20:80, 20:180] = 1
    cv2.imwrite(str(tmp_path / "regions.png"), regions)
    pieces = cut(regions, [(100, 10), (100, 90)], next_id=2, min_piece=50)
    record = {"line": [[100, 10], [100, 90]], "min_piece": 50, "pieces": {str(k): v for k, v in pieces.items()}}
    (tmp_path / "cuts.json").write_text(json.dumps({"cuts": [record], "next_id": 4}))
    replayed, cuts = load_regions(tmp_path, min_piece=999)  # replays with the recorded min_piece
    assert (replayed == regions).all() and len(cuts) == 1


def test_neighbors_reach_across_walls_only():
    regions = two_regions(6)
    regions[90:98, 20:40] = 3   # 10 px below region 1
    assert neighbors(regions, 8) == {1: [2], 2: [1], 3: []}


# --- drawn walls (labeler W) ---

def box_building(y0=20, y1=80, x0=20, x1=100, b=1, shape=(120, 160)):
    inst = np.zeros(shape, np.int32)
    inst[y0:y1, x0:x1] = b
    return inst


def test_wall_across_spill_removes_the_smaller_side():
    inst = box_building()                                   # real wall at x = 90: x 90..99 is spill
    assert apply_wall(inst, 1, [(89, 10), (89, 90)]) > 0
    assert (inst[20:80, 20:88] == 1).all() and not inst[:, 91:].any()
    assert not inst[:20].any() and not inst[80:].any()     # the line runs past the corners: no nubs there


def test_flip_keeps_the_other_side():
    inst = box_building()
    apply_wall(inst, 1, [(89, 10), (89, 90)], flip=True)
    assert not inst[:, :88].any() and (inst[20:80, 91:100] == 1).all()


def test_wall_fills_a_missing_corner():
    inst = box_building()
    inst[20:40, 80:100] = 0                                 # top right corner missing
    # from the top edge out to the real corner and down to the right edge (ends a few px off: snapped)
    apply_wall(inst, 1, [(76, 21), (99, 21), (99, 44)])
    assert (inst[22:80, 22:99] == 1).all()


def test_wall_extends_an_edge_that_stops_short():
    inst = box_building(x1=90)                              # real wall at x = 99
    apply_wall(inst, 1, [(88, 20), (99, 20), (99, 79), (88, 79)])
    assert (inst[20:80, 20:100] == 1).all() and not inst[:, 101:].any()


def test_wall_never_takes_another_building():
    inst = box_building(x1=90)
    inst[20:80, 92:120] = 2                                 # a neighbor where the line would grow
    before = inst == 2
    apply_wall(inst, 1, [(88, 20), (110, 20), (110, 79), (88, 79)])
    assert ((inst == 2) == before).all()


def test_line_off_the_building_does_nothing():
    inst = box_building()
    ref = inst.copy()
    assert apply_wall(inst, 1, [(130, 10), (130, 100)]) == 0 and (inst == ref).all()
    inst[20:40, 80:100] = 0
    ref = inst.copy()
    assert apply_wall(inst, 1, [(76, 21), (99, 21), (99, 44)], flip=True) == 0   # only fills: nothing to flip
    assert (inst == ref).all()


def test_walls_follow_their_region_and_skip_dropped_ones():
    inst = box_building()
    done = apply_walls(inst, [{"line": [(89, 10), (89, 90)], "region": 5}, {"line": [(50, 0), (50, 99)], "region": 9}],
                       {1: [4, 5]})
    assert done[0] > 0 and done[1] is None and not inst[:, 91:].any()


def test_dissolve_applies_walls_after_the_fit(run_dissolve):
    regions = np.zeros((120, 160), np.uint16)
    regions[20:80, 20:100] = 1
    inst, valid, info = run_dissolve(regions, {"walls": [{"line": [(89, 10), (89, 90)], "region": 1}]}, CFG)
    assert info["walls"] == 1 and not info["walls_skipped"]
    assert inst[50, 60] == 1 and not inst[:, 91:].any()
