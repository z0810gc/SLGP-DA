# -*- coding: utf-8 -*-
"""
Axial stepwise brightening + dynamic ADD(A-B) + backfill replacement
- A: median value after eroding the original cavity mask inward by 3 px
- B: median value of the new region after shifting the shrunken mask 10 steps along the steel-anchor main axis (intersected with steel & half-space)
- ADD = A - B; during enhancement: "brighten first, then blend with feathered alpha"
- Direction compatible: GROW_DIRECTION = "right"/"left"
- Outputs include: histograms, results with/without annotations, backfill images, etc.
"""

from ultralytics import YOLO
import numpy as np
import cv2
import os

# Optional: plot histograms
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ========= Parameters =========
# GROW_DIRECTION    = "right"   # "right" or "left"
GROW_DIRECTION    = "left"   # "right" or "left"
STEP_PX           = 8.0      # Pixels translated along the main axis per step
NUM_STEPS         = 25       # Number of steps
GAIN              = 1.01     # Typically keep as 1.00
ADD               = 0.0      # Fallback only; actual uses A-B
EXTRA_BRIGHTEN    = -5        # Manually added pixel brightening
FEATHER_PX        = 8        # Feather radius
ONLY_NEW_AREA     = False    # Brighten newly added region only
LINE_THICKNESS    = 1        # Main-axis line thickness
SAVE_EVERY_STEP   = False    # Whether to save each step result

BRIGHTEN_FIRST_OVERLAP = True  # Whether to pre-brighten the first-step overlap region
OVERLAP_DILATE_PX      = 5     # Overlap dilation
OVERLAP_FEATHER_PX     = 8.0   # Overlap feather

CAVITY_OUTLINE_COLOR   = (0, 0, 255)     # Red
CAVITY_OUTLINE_THICK   = 1              # Outline thickness
CAVITY_OUTLINE_LINE_TYPE = cv2.LINE_AA  # Outline line type

REPLACE_STEPS = 3   # Backfill translation steps

# Parameters for A/B scheme
ERODE_PX        = 5   # Erosion radius
SHIFT_STEPS_AB  = 10  # Steps used to compute B
USE_MEDIAN      = True

# Classes and paths
STEEL_ID, CAV_ID = 0, 1   # Class IDs, following the training order
MODEL_WTS = "/home/zgc/datawork/DRimage/ultralytics-main/runs/segment/train4/weights/best.pt"
IMG_PATH  = "/home/zgc/datawork/DRimage/XDR_DATA_cleaned/271.png"
# IMG_PATH  = "/home/zgc/datawork/DRimage/ultralytics-main/runs/segment/predict53/image_cavity_shift_steps.png"


# ---------------- Utility functions ----------------
def largest_component_mask(bin_mask: np.ndarray) -> np.uint8:
    bin_mask = (bin_mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(bin_mask)
    if num_labels <= 1:
        return bin_mask
    areas = [(labels == i).sum() for i in range(1, num_labels)]
    biggest = 1 + int(np.argmax(areas))
    return (labels == biggest).astype(np.uint8)


def pca_axis_from_mask(bin_mask: np.ndarray):
    ys, xs = np.where(bin_mask > 0)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    c = pts.mean(axis=0, keepdims=True)
    pts_c = pts - c
    cov = (pts_c.T @ pts_c) / max(len(pts_c) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, np.argmax(evals)]
    v = v / (np.linalg.norm(v) + 1e-12)
    return c[0], v  # (cx,cy), (vx,vy)


def clip_line_to_mask(c, v, bin_mask):
    ys, xs = np.where(bin_mask > 0)
    if xs.size < 2:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64)
    t = (pts - c).dot(v)
    t_min, t_max = float(np.min(t)), float(np.max(t))
    p1 = c + t_min * v
    p2 = c + t_max * v
    return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))


def draw_segment(img_bgr, p1, p2, color=(0, 255, 0), thickness=2):
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    x2, y2 = int(round(p2[0])), int(round(p2[1]))
    cv2.line(img_bgr, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    cv2.circle(img_bgr, (x1, y1), 3, (0, 0, 255), -1)
    cv2.circle(img_bgr, (x2, y2), 3, (255, 0, 0), -1)


def to_gray(img_bgr_uint8: np.ndarray) -> np.ndarray:
    if img_bgr_uint8.ndim == 2:
        return img_bgr_uint8.copy()
    return cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2GRAY)


def plot_histogram(valuesA, valuesB, save_dir, A_med, B_med):
    if not HAS_MPL:
        print("[Warn] matplotlib not available, skipping histogram saving")
        return
    # A
    plt.figure(figsize=(6, 4))
    plt.hist(valuesA, bins=256, range=(0, 255), alpha=0.9)
    plt.axvline(A_med, linestyle="--")
    plt.title(f"A: Shrink@3px median={A_med:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hist_A_shrink.png"), dpi=140)
    plt.close()
    # B
    plt.figure(figsize=(6, 4))
    plt.hist(valuesB, bins=256, range=(0, 255), alpha=0.9)
    plt.axvline(B_med, linestyle="--")
    plt.title(f"B: Shift@{SHIFT_STEPS_AB}steps median={B_med:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hist_B_shift10.png"), dpi=140)
    plt.close()
    # Comparison
    plt.figure(figsize=(7, 4))
    plt.hist(valuesA, bins=256, range=(0, 255), alpha=0.5, label=f"A median={A_med:.2f}")
    plt.hist(valuesB, bins=256, range=(0, 255), alpha=0.5, label=f"K median={B_med:.2f}")
    plt.axvline(A_med, color="C0", linestyle="--")
    plt.axvline(B_med, color="C1", linestyle="--")
    plt.legend()
    plt.title("Histogram: A (shrink) vs K (shift 15 steps)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hist_compare_A_vs_B.png"), dpi=160)
    plt.close()


# ---------------- Main pipeline ----------------
model = YOLO(MODEL_WTS)
results = model(IMG_PATH, save=True, conf=0.25, iou=0.5)

for result in results:
    print("=" * 60)
    H, W = result.orig_shape

    # === Placeholders for three-stage visualization ===
    vis_step1 = None  # Axial prior + half-space constraint
    vis_step2 = None  # Axial stepwise cavity-mask expansion
    vis_step3 = None  # Brightness alignment + boundary feather blending

    # Original image
    base = (
        result.orig_img.copy()
        if hasattr(result, "orig_img") and result.orig_img is not None
        else cv2.imread(IMG_PATH)
    )
    if base is None:
        print("[Err] Failed to read the original image")
        break
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    # YOLO masks
    if result.masks is None or result.boxes is None:
        print("[Warn] No segmentation or detection boxes")
        break
    masks_t = result.masks.data
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)

    steel = np.zeros((H, W), np.uint8)
    cav_raw = np.zeros((H, W), np.uint8)
    for i in range(len(masks_t)):
        m = (masks_t[i].cpu().numpy() > 0.5).astype(np.uint8)
        m_up = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        if cls_ids[i] == STEEL_ID:
            steel |= m_up
        if cls_ids[i] == CAV_ID:
            cav_raw |= m_up

    steel = largest_component_mask(steel)
    cav = largest_component_mask(cav_raw) if cav_raw.sum() > 0 else cav_raw

    save_dir = getattr(result, "save_dir", None) or "runs/segment/predict"
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, "steel_mask.png"), steel * 255)
    if cav.sum() > 0:
        cv2.imwrite(os.path.join(save_dir, "cavity_mask_used.png"), cav * 255)

    if steel.sum() < 10 or cav.sum() < 10:
        print("[Warn] Too few pixels in steel or cavity mask")
        break

    # Main axis (unified to the right: v.x >= 0)
    c, v = pca_axis_from_mask(steel)
    if v[0] < 0:
        v = -v
    seg = clip_line_to_mask(c, v, steel)
    if seg is None:
        print("[Warn] Failed to obtain the main-axis segment")
        break
    p1, p2 = seg

    # Direction
    DIR_SIGN = 1 if str(GROW_DIRECTION).lower() == "right" else -1

    # Half-space (split by cavity centroid)
    ys0, xs0 = np.where(cav > 0)
    cx0, cy0 = xs0.mean(), ys0.mean()
    X, Y = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)
    )
    proj = (X - cx0) * v[0] + (Y - cy0) * v[1]
    half_space = (proj >= 0) if DIR_SIGN > 0 else (proj <= 0)
    steel_bool = steel > 0

    # === Step1: Axial prior + half-space constraint visualization ===
    vis_step1 = base.copy()

    # Slightly blue tint within steel region to highlight steel-anchor extent
    steel_region = steel_bool
    vis_step1[steel_region] = (
        0.6 * vis_step1[steel_region]
        + 0.4 * np.array([180, 180, 255], dtype=np.uint8)
    ).astype(np.uint8)

    # Red tint within original cavity region to highlight initial cavity
    cav_region = cav > 0
    vis_step1[cav_region] = (
        0.5 * vis_step1[cav_region]
        + 0.5 * np.array([0, 0, 255], dtype=np.uint8)
    ).astype(np.uint8)

    # Slightly darken steel region outside the half-space to indicate "excluded half-space"
    outside_hs = steel_bool & (~half_space)
    vis_step1[outside_hs] = (0.4 * vis_step1[outside_hs]).astype(np.uint8)

    # Draw PCA main axis
    draw_segment(vis_step1, p1, p2, color=(0, 255, 0), thickness=2)

    # Top-left annotation
    cv2.putText(
        vis_step1,
        "",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis_step1,
        "",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # === Dynamic ADD via A-B ===
    gray = to_gray(base)

    # A) Erode cav
    ker = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ERODE_PX + 1, 2 * ERODE_PX + 1)
    )
    cav_shrink = (
        cv2.erode((cav * 255).astype(np.uint8), ker, iterations=1) > 0
    )

    # B) Shift cav_shrink 10 steps along the main axis, and intersect with steel & half-space
    shift_offset = float(SHIFT_STEPS_AB * STEP_PX * DIR_SIGN)
    mapx = (X - v[0] * shift_offset).astype(np.float32)
    mapy = (Y - v[1] * shift_offset).astype(np.float32)
    cav_shrink_shift = (
        cv2.remap(
            (cav_shrink.astype(np.uint8) * 255),
            mapx,
            mapy,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )
    new_region_mask = cav_shrink_shift & steel_bool & half_space

    valsA = gray[cav_shrink]
    valsB = gray[new_region_mask]

    if valsA.size == 0 or valsB.size == 0:
        print(
            f"[Warn] Insufficient pixels in A/B regions: A={valsA.size}, B={valsB.size}, fallback to fixed ADD={ADD}"
        )
        ADD_dynamic = float(ADD)
    else:
        A_med = float(np.median(valsA)) if USE_MEDIAN else float(np.mean(valsA))
        B_med = float(np.median(valsB)) if USE_MEDIAN else float(np.mean(valsB))
        ADD_dynamic = A_med - B_med + EXTRA_BRIGHTEN
        print(
            f"[Info] A_med={A_med:.2f}, B_med={B_med:.2f}, ADD=A-B={ADD_dynamic:.2f}"
        )
        try:
            plot_histogram(valsA, valsB, save_dir, A_med, B_med)
        except Exception as e:
            print(f"[Warn] Failed to save histograms: {e}")

    # Initialize output
    out = base.copy().astype(np.float32)
    union_mask = np.zeros((H, W), np.uint8)

    # ===== Pre-brighten first-step overlap (brighten first, then alpha-feather blend) =====
    if BRIGHTEN_FIRST_OVERLAP and NUM_STEPS >= 1 and STEP_PX > 0:
        dx1 = float(v[0] * STEP_PX * DIR_SIGN)
        dy1 = float(v[1] * STEP_PX * DIR_SIGN)
        j1 = (X - dx1).astype(np.float32)
        i1 = (Y - dy1).astype(np.float32)
        cav_step1 = (
            cv2.remap(
                (cav * 255).astype(np.uint8),
                j1,
                i1,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        )
        step1_mask = (cav_step1 & steel_bool & half_space).astype(np.uint8)
        overlap = (step1_mask & cav.astype(np.uint8)).astype(np.uint8)

        if overlap.sum() > 0:
            if OVERLAP_DILATE_PX > 0:
                ker_ov = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * OVERLAP_DILATE_PX + 1, 2 * OVERLAP_DILATE_PX + 1),
                )
                overlap = cv2.dilate(overlap, ker_ov, iterations=1)

            if OVERLAP_FEATHER_PX > 0:
                dist = cv2.distanceTransform(
                    overlap * 255, cv2.DIST_L2, 3
                ).astype(np.float32)
                alpha_ov = (
                    np.clip(
                        dist / float(OVERLAP_FEATHER_PX),
                        0.0,
                        1.0,
                    )
                    * overlap.astype(np.float32)
                )
            else:
                alpha_ov = overlap.astype(np.float32)

            for ch in range(3):
                src = out[..., ch]
                bright = np.clip(src * GAIN + ADD_dynamic, 0, 255)
                out[..., ch] = alpha_ov * bright + (1.0 - alpha_ov) * src

            union_mask = np.maximum(union_mask, overlap)

    # ===== Stepping loop (brighten first -> then feather blend) =====
    for k in range(1, NUM_STEPS + 1):
        offset = k * STEP_PX
        dx = float(v[0] * offset * DIR_SIGN)
        dy = float(v[1] * offset * DIR_SIGN)

        j_src = (X - dx).astype(np.float32)
        i_src = (Y - dy).astype(np.float32)
        cav_shift = (
            cv2.remap(
                (cav * 255).astype(np.uint8),
                j_src,
                i_src,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        )

        step_mask = (cav_shift & steel_bool & half_space).astype(np.uint8)

        if ONLY_NEW_AREA:
            apply_mask = (step_mask & (1 - union_mask)).astype(np.uint8)
        else:
            apply_mask = step_mask

        if apply_mask.sum() == 0:
            if SAVE_EVERY_STEP:
                cv2.imwrite(
                    os.path.join(save_dir, f"_step{k:02d}_apply_zero.png"),
                    apply_mask * 255,
                )
            continue

        if FEATHER_PX > 0:
            dist = cv2.distanceTransform(
                apply_mask * 255, cv2.DIST_L2, 3
            ).astype(np.float32)
            alpha = (
                np.clip(dist / float(FEATHER_PX), 0.0, 1.0)
                * apply_mask.astype(np.float32)
            )
        else:
            alpha = apply_mask.astype(np.float32)

        for ch in range(3):
            src = out[..., ch]
            bright = np.clip(src * GAIN + ADD_dynamic, 0, 255)
            out[..., ch] = alpha * bright + (1.0 - alpha) * src

        union_mask = ((union_mask > 0) | (apply_mask > 0)).astype(np.uint8)

        if SAVE_EVERY_STEP:
            tmp = np.clip(out, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, f"_step{k:02d}_out.png"), tmp)
            cv2.imwrite(
                os.path.join(save_dir, f"_step{k:02d}_mask.png"), apply_mask * 255
            )

    # === Step2: Axial stepwise cavity-mask expansion visualization (union_mask of all steps) ===
    vis_step2 = base.copy()
    grow_region = union_mask > 0
    vis_step2[grow_region] = (
        0.5 * vis_step2[grow_region]
        + 0.5 * np.array([0, 165, 255], dtype=np.uint8)
    ).astype(np.uint8)  # Orange highlight
    draw_segment(vis_step2, p1, p2, color=(0, 255, 0), thickness=2)
    cv2.putText(
        vis_step2,
        "",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis_step2,
        "",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # ===== Final output and overlay =====
    out = np.clip(out, 0, 255).astype(np.uint8)

    overlay = out.copy()
    edge = cv2.morphologyEx(
        union_mask,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ) > 0
    overlay[edge] = (
        0.5 * overlay[edge] + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)
    draw_segment(overlay, p1, p2, (0, 255, 0), LINE_THICKNESS)

    # —— Original cavity red outline (contour) ——
    cav_u8 = cav.astype(np.uint8) * 255
    contours_orig, _ = cv2.findContours(
        cav_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    out_red = out.copy()
    overlay_red = overlay.copy()
    if contours_orig:
        cv2.drawContours(
            out_red,
            contours_orig,
            -1,
            CAVITY_OUTLINE_COLOR,
            CAVITY_OUTLINE_THICK,
            lineType=CAVITY_OUTLINE_LINE_TYPE,
        )
        cv2.drawContours(
            overlay_red,
            contours_orig,
            -1,
            CAVITY_OUTLINE_COLOR,
            CAVITY_OUTLINE_THICK,
            lineType=CAVITY_OUTLINE_LINE_TYPE,
        )

    # === Step3: Brightness alignment + boundary feather blending visualization ===
    vis_step3 = overlay_red.copy()
    cv2.putText(
        vis_step3,
        "",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis_step3,
        "",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # ===== Backfill replacement (direction compatible) =====
    total_offset = float(REPLACE_STEPS * STEP_PX)
    dx_tot = float(v[0] * total_offset * DIR_SIGN)
    dy_tot = float(v[1] * total_offset * DIR_SIGN)

    # Source mask: shift cav by ±(dx_tot, dy_tot)
    mapx2 = (X - dx_tot).astype(np.float32)
    mapy2 = (Y - dy_tot).astype(np.float32)
    cav_src = (
        cv2.remap(
            (cav * 255).astype(np.uint8),
            mapx2,
            mapy2,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )
    cav_src_u8 = cav_src.astype(np.uint8) * 255

    # Source pixels: shift in the same direction as the source mask (key fix)
    mapx3 = (X + dx_tot).astype(np.float32)
    mapy3 = (Y + dy_tot).astype(np.float32)
    out_src_for_dest = cv2.remap(
        out,
        mapx3,
        mapy3,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE,
    )

    mask2d = cav > 0
    out_replaced = out.copy()
    overlay_replaced = overlay.copy()
    out_replaced[mask2d] = out_src_for_dest[mask2d]
    overlay_replaced[mask2d] = out_src_for_dest[mask2d]

    if contours_orig:
        cv2.drawContours(
            out_replaced,
            contours_orig,
            -1,
            CAVITY_OUTLINE_COLOR,
            CAVITY_OUTLINE_THICK,
            lineType=CAVITY_OUTLINE_LINE_TYPE,
        )
        cv2.drawContours(
            overlay_replaced,
            contours_orig,
            -1,
            CAVITY_OUTLINE_COLOR,
            CAVITY_OUTLINE_THICK,
            lineType=CAVITY_OUTLINE_LINE_TYPE,
        )

    contours_src, _ = cv2.findContours(
        cav_src_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours_src:
        cv2.drawContours(
            out_replaced,
            contours_src,
            -1,
            CAVITY_OUTLINE_COLOR,
            CAVITY_OUTLINE_THICK,
            lineType=CAVITY_OUTLINE_LINE_TYPE,
        )
        cv2.drawContours(
            overlay_replaced,
            contours_src,
            -1,
            CAVITY_OUTLINE_COLOR,
            CAVITY_OUTLINE_THICK,
            lineType=CAVITY_OUTLINE_LINE_TYPE,
        )

    # Clean backfill image
    out_replaced_clean = out.copy()
    out_replaced_clean[mask2d] = out_src_for_dest[mask2d]

    # ===== Save =====
    out_path = os.path.join(save_dir, "image_cavity_shift_steps.png")
    ov_path = os.path.join(save_dir, "overlay_cavity_shift_steps.png")
    out_path_red = os.path.join(save_dir, "image_cavity_shift_steps_red.png")
    ov_path_red = os.path.join(save_dir, "overlay_cavity_shift_steps_red.png")
    out_path_replaced = os.path.join(save_dir, "image_cavity_replaced.png")
    ov_path_replaced = os.path.join(save_dir, "overlay_cavity_replaced.png")
    clean_path = os.path.join(save_dir, "image_cavity_replaced_clean.png")

    cv2.imwrite(out_path, out)
    cv2.imwrite(ov_path, overlay)
    cv2.imwrite(out_path_red, out_red)
    cv2.imwrite(ov_path_red, overlay_red)
    cv2.imwrite(out_path_replaced, out_replaced)
    cv2.imwrite(ov_path_replaced, overlay_replaced)
    cv2.imwrite(clean_path, out_replaced_clean)

    # ===== Save three-stage pipeline figures =====
    if vis_step1 is not None and vis_step2 is not None and vis_step3 is not None:
        h0, w0 = vis_step1.shape[:2]
        vis2_resized = cv2.resize(vis_step2, (w0, h0), interpolation=cv2.INTER_AREA)
        vis3_resized = cv2.resize(vis_step3, (w0, h0), interpolation=cv2.INTER_AREA)
        pipeline_img = np.concatenate(
            [vis_step1, vis2_resized, vis3_resized], axis=1
        )

        cv2.imwrite(
            os.path.join(save_dir, "gppl_step1_axis_halfspace.png"), vis_step1
        )
        cv2.imwrite(
            os.path.join(save_dir, "gppl_step2_axial_growth.png"), vis2_resized
        )
        cv2.imwrite(
            os.path.join(save_dir, "gppl_step3_brighten_feather.png"),
            vis3_resized,
        )
        cv2.imwrite(
            os.path.join(save_dir, "gppl_3steps_pipeline.png"), pipeline_img
        )

    print(
        f"[Info] dir={GROW_DIRECTION}, step_px={STEP_PX}, num_steps={NUM_STEPS}, "
        f"grow_offset≈{STEP_PX*NUM_STEPS:.1f}px, replace_steps={REPLACE_STEPS}, "
        f"AB_shift_steps={SHIFT_STEPS_AB}, ADD_dynamic={ADD_dynamic:.2f}"
    )
    print(f"[Save] Result: {out_path}")
    print(f"[Save] Overlay: {ov_path}")
    print(f"[Save] Result (red outline): {out_path_red}")
    print(f"[Save] Overlay (red outline): {ov_path_red}")
    print(f"[Save] Backfill (with source/original red outlines): {out_path_replaced}")
    print(f"[Save] Overlay backfill (with source/original red outlines): {ov_path_replaced}")
    print(f"[Save] Clean backfill result: {clean_path}")
    print(
        "[Save] Histograms: hist_A_shrink.png, hist_B_shift10.png, "
        "hist_compare_A_vs_B.png"
    )
    print(
        "[Save] Three-stage pipeline: gppl_step1_axis_halfspace.png, "
        "gppl_step2_axial_growth.png, gppl_step3_brighten_feather.png, "
        "gppl_3steps_pipeline.png"
    )
