"""Metrics: PQ, recall per size bucket, merges/splits, stain false positives, under-segmentation (guide §7, §9.2)."""
import numpy as np


def overseg_checks(gt, pred, cover=0.2):
    """The two over-segmenter rules from §9.2, given a labeled instance map `gt` and region map `pred`.

    Rule 1: a region merges buildings g and g' if it covers at least `cover` of each. The
    under-segmentation rate is the fraction of buildings merged with another; it should be ~0.
    Rule 2: regions per building counts regions whose majority lies in that building; aim for 1.5-3.
    """
    n_gt, n_pred = int(gt.max()), int(pred.max())
    pair = gt.ravel().astype(np.int64) * (n_pred + 1) + pred.ravel()
    ov = np.bincount(pair, minlength=(n_gt + 1) * (n_pred + 1)).reshape(n_gt + 1, n_pred + 1)
    O = ov[1:, 1:]                                  # building x region overlap in px
    gt_area, pred_area = ov[1:].sum(1), ov[:, 1:].sum(0)

    covers = O >= cover * gt_area[:, None]
    merged = covers[:, covers.sum(0) >= 2].any(1)
    major = O.max(0) > 0.5 * pred_area
    per_building = np.bincount(O.argmax(0)[major], minlength=n_gt)
    return {
        "buildings": n_gt,
        "regions": n_pred,
        "under_segmented": int(merged.sum()),
        "under_seg_rate": round(float(merged.mean()), 4),
        "buildings_with_region": int((per_building > 0).sum()),
        "regions_per_building_mean": round(float(per_building.mean()), 2),
        "regions_per_building_median": float(np.median(per_building)),
        "regions_mostly_outside": int((O.sum(0) < 0.5 * pred_area).sum()),
    }
