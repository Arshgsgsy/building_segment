"""Over-segmenter metrics (guide §9.2)."""
import numpy as np


def overseg_checks(gt, pred, cover=0.2):
    """Under-segmented buildings and regions per building, for instance map `gt` and region map `pred`."""
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
