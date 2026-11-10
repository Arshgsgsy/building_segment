"""Dissolve-time wall fit (labels.fit_to_walls): outlines onto the drawn walls, no pixel between buildings."""
import numpy as np

from sanborn import load_config
from sanborn.labels import dash_lines, dissolve, fit_to_walls

CFG = load_config()["labels"]


def fit(inst, masks):
    return fit_to_walls(inst, masks, max_hole=CFG["max_hole"], **CFG["fit"])


def masks_for(shape, ink=None, ink_all=None, fill=None):
    """Masks as wall_masks gives them: ink is dark (L* 40), the rest paper (L* 97)."""
    z = np.zeros(shape, bool)
    ink = z if ink is None else ink
    ink_all = ink | (z if ink_all is None else ink_all)
    return {"ink": ink, "ink_all": ink_all, "fill": z if fill is None else fill,
            "L": np.where(ink_all, 40, 97).astype(np.uint8)}


def walled_box(shape=(100, 100), y0=20, y1=80, x0=20, x1=80, t=4):
    """Ink of a 4 px wall ring, and the fill inside it."""
    ink = np.zeros(shape, bool)
    ink[y0:y1, x0:x1] = True
    ink[y0 + t:y1 - t, x0 + t:x1 - t] = False
    fill = np.zeros(shape, bool)
    fill[y0 + t:y1 - t, x0 + t:x1 - t] = True
    return ink, fill


def test_outer_wall_is_claimed_to_its_outer_edge_and_not_beyond():
    ink, fill = walled_box()
    ink[50, 0:20] = True                        # a lot line touching the wall from outside
    inst = np.zeros((100, 100), np.int32)
    inst[22:78, 22:78] = 1                      # label stops 2 px inside the wall
    out = fit(inst, masks_for(inst.shape, ink, fill=fill))
    assert (out[20:80, 20:80] == 1).all()       # whole wall covered
    assert not out[:20].any() and not out[80:].any() and not out[:, 80:].any()
    assert not out[50, :18].any()               # lot line not claimed


def test_spill_past_a_dashed_side_is_removed():
    ink, fill = walled_box()
    ink[76:80, 24:76] = False                   # bottom side drawn dashed instead
    dash = np.zeros((100, 100), bool)
    for x in range(24, 76, 12):
        dash[77:80, x:x + 7] = True
    fill[76:86, 24:76] = True                   # the color wash runs 6 px past the dashes
    inst = np.zeros((100, 100), np.int32)
    inst[20:88, 20:80] = 1                      # and the over-segmenter grew 2 px further
    m = masks_for(inst.shape, ink, dash, fill)
    assert dash_lines(m["ink_all"], m["ink"])[77:80, 24:76].all()  # the dashes become one line
    out = fit(inst, m)
    assert not out[81:].any()                   # nothing past the dashed line
    assert (out[20:80, 20:80] == 1).all()       # dashes and the gaps between them are inside


def test_wedge_cut_off_by_text_touching_the_wall_is_given_back():
    ink, fill = walled_box()
    stroke = np.zeros((100, 100), bool)         # a "1" from the top wall to the left wall
    for i in range(18):
        stroke[24 + i, 41 - i:44 - i] = True
    ink |= stroke
    inst = np.zeros((100, 100), np.int32)
    inst[24:76, 24:76] = 1
    yy, xx = np.mgrid[:100, :100]
    wedge = (yy - 24) + (xx - 24) < 17          # the corner behind the stroke: the over-segmenter's notch
    inst[wedge | stroke] = 0
    out = fit(inst, masks_for(inst.shape, ink, fill=fill & ~stroke))
    assert (out[20:80, 20:80] == 1).all()       # wedge, stroke and walls are all the building
    assert not out[:20].any() and not out[:, :20].any()


def test_open_strip_behind_a_broken_wall_is_not_taken():
    shape = (100, 140)
    ink, fill = walled_box(shape)
    ink[24:76, 76:80] = False                   # right wall: heavy dashes 25 px long, 10 px gaps
    for y in (20, 55):
        ink[y:y + 25, 76:80] = True
    ink[16:20, 20:112] = ink[80:84, 20:112] = True  # walls above and below run on to a lot line
    ink[16:84, 108:112] = True
    fill[24:76, 84:94] = True                   # a colored strip the over-segmenter dropped (a gallery)
    inst = np.zeros(shape, np.int32)
    inst[24:76, 24:76] = 1
    m = masks_for(shape, ink, fill=fill)
    for cap in (CFG["fit"]["cell_piece"], 10 ** 9):  # the size cap alone, then the enclosed-piece rule alone
        out = fit_to_walls(inst, m, max_hole=CFG["max_hole"], **{**CFG["fit"], "cell_piece": cap})
        assert not (out[:, 84:] == 1).any()     # the strip stays out: it is open to paper on its far side
        assert (out[20:80, 20:76] == 1).all()


def test_double_wall_keeps_its_contact():
    regions = np.zeros((100, 220), np.uint16)
    regions[20:80, 20:99] = 1                   # two buildings whose regions met in the paper gap
    regions[20:80, 99:180] = 2                  # of a double wall (walls at 94-97 and 101-104)
    ink = np.zeros(regions.shape, bool)
    ink[20:80, 94:97] = ink[20:80, 101:104] = True
    inst, _, info = dissolve(regions, {}, CFG, masks_for(regions.shape, ink, fill=(regions > 0) & ~ink))
    assert info["party_walls"] == 1
    assert (inst[20:80, 94:104] > 0).all()      # walls and the gap between them stay labeled


def test_no_pixel_changes_building():
    regions = np.zeros((100, 220), np.uint16)
    regions[20:80, 20:96] = 1
    regions[20:80, 102:178] = 2                 # 6 px of shared wall ink nobody claimed
    ink = np.zeros(regions.shape, bool)
    ink[18:82, 96:102] = True
    inst = regions.astype(np.int32)
    out = fit(inst, masks_for(regions.shape, ink, fill=regions > 0))
    assert (out[20:80, 96:99] == 1).all() and (out[20:80, 99:102] == 2).all()  # split down the middle
    assert not ((inst > 0) & (out > 0) & (out != inst)).any()


def test_text_row_is_not_a_dashed_line():
    ink_all = np.zeros((100, 200), bool)
    for x in range(20, 180, 12):
        ink_all[30:33, x:x + 7] = True          # dashes
        ink_all[60:74, x:x + 9] = True          # letters
    out = dash_lines(ink_all, np.zeros_like(ink_all))
    assert out[30:33, 20:175].all() and not out[50:].any()
