import os
import glob
import csv
import argparse
import re

import cv2
import numpy as np


# ---------- Basic utility functions, same as the previous script ----------

def load_gray_image(path):
    """Read an image in grayscale mode and return a np.uint8 array."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def build_boundary_bands(mask, radius=3):
    """
    Generate inner/outer narrow boundary bands from a binary mask:
    - mask: 0/255 or 0/1 mask, where 1 indicates the defect region
    - radius: morphological kernel radius, approximately equal to the band width (pixels)
    Returns:
    - Omega_in: inner band (uint8, 0/1)
    - Omega_out: outer band (uint8, 0/1)
    """
    mask01 = (mask > 0).astype(np.uint8)

    ksize = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    eroded = cv2.erode(mask01, kernel)
    omega_in = mask01 - eroded

    dilated = cv2.dilate(mask01, kernel)
    omega_out = dilated - mask01

    return omega_in, omega_out


def masked_mean(arr, mask_band):
    """Compute the mean of arr within the region where mask_band==1."""
    m = mask_band.astype(bool)
    if m.sum() == 0:
        return np.nan
    return float(arr[m].mean())


def rms_contrast(intensity, mask_band):
    """
    Compute RMS contrast within the region where mask_band==1:
    C_RMS = std / mean
    """
    m = mask_band.astype(bool)
    vals = intensity[m].astype(np.float32)
    if vals.size == 0:
        return np.nan
    mean = vals.mean()
    std = vals.std()
    return float(std / (mean + 1e-6))


def compute_metrics(img, mask, radius=3):
    """
    Compute the following for an image and its corresponding mask:
    - Gradient magnitude difference ΔG
    - RMS contrast difference ΔC
    """
    omega_in, omega_out = build_boundary_bands(mask, radius=radius)

    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    G = cv2.magnitude(gx, gy)

    mu_g_in = masked_mean(G, omega_in)
    mu_g_out = masked_mean(G, omega_out)
    delta_G = abs(mu_g_in - mu_g_out)

    C_in = rms_contrast(img, omega_in)
    C_out = rms_contrast(img, omega_out)
    delta_C = abs(C_in - C_out)

    return {
        "mu_g_in": mu_g_in,
        "mu_g_out": mu_g_out,
        "Delta_G": delta_G,
        "C_in": C_in,
        "C_out": C_out,
        "Delta_C": delta_C,
    }


# ---------- Parse command-line arguments ----------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch evaluate ΔG / ΔC for clean & SLMY images "
                    "using masks with specific naming patterns."
    )
    parser.add_argument(
        "--masks_dir", type=str, required=True,
        help="Directory of masks, e.g. 1_mask_used.png, 2_mask_used.png ..."
    )
    parser.add_argument(
        "--clean_dir", type=str, required=True,
        help="Directory of clean images, e.g. 1.png, 2.png ..."
    )
    parser.add_argument(
        "--gen_dir", type=str, required=True,
        help="Directory of SLMY generated images, "
             "e.g. 1_xxx.png, 2_xxx.png ..."
    )
    parser.add_argument(
        "--out_csv", type=str, required=True,
        help="Output CSV path."
    )
    parser.add_argument(
        "--band_radius", type=int, default=3,
        help="Radius (in pixels) for boundary bands."
    )
    return parser.parse_args()


# ---------- Main logic ----------

def main():
    args = parse_args()
    masks_dir = args.masks_dir
    clean_dir = args.clean_dir
    gen_dir = args.gen_dir
    out_csv = args.out_csv
    radius = args.band_radius

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # 1) Build mask dict: id -> mask_path
    mask_paths = [
        p for p in glob.glob(os.path.join(masks_dir, "*"))
        if os.path.splitext(p)[1].lower() in exts
    ]
    mask_index = {}
    id_pattern = re.compile(r"^(\d+)")  # Leading digits in filename

    for mp in mask_paths:
        fname = os.path.basename(mp)  # e.g. "1_mask_used.png"
        m = id_pattern.match(fname)
        if not m:
            print(f"[WARN] skip mask without leading id: {fname}")
            continue
        img_id = m.group(1)
        mask_index[img_id] = mp

    print(f"Found {len(mask_index)} mask ids in {masks_dir}")

    rows = []

    # 2) Process clean directory: filenames like "1.png"
    clean_paths = [
        p for p in glob.glob(os.path.join(clean_dir, "*"))
        if os.path.splitext(p)[1].lower() in exts
    ]
    clean_paths.sort()

    print(f"Found {len(clean_paths)} clean images in {clean_dir}")

    for cp in clean_paths:
        fname = os.path.basename(cp)  # e.g. "1.png"
        m = id_pattern.match(fname)
        if not m:
            print(f"[WARN] skip clean image without leading id: {fname}")
            continue
        img_id = m.group(1)

        if img_id not in mask_index:
            print(f"[WARN] no mask for clean id={img_id}")
            continue

        mask_path = mask_index[img_id]

        try:
            img = load_gray_image(cp)
            mask = load_gray_image(mask_path)
            metrics = compute_metrics(img, mask, radius=radius)

            rows.append({
                "id": img_id,
                "group": "clean",
                "img_file": fname,
                "mask_file": os.path.basename(mask_path),
                "mu_g_in": metrics["mu_g_in"],
                "mu_g_out": metrics["mu_g_out"],
                "Delta_G": metrics["Delta_G"],
                "C_in": metrics["C_in"],
                "C_out": metrics["C_out"],
                "Delta_C": metrics["Delta_C"],
            })
        except Exception as e:
            print(f"[ERROR] clean id={img_id}, file={fname}: {e}")

    # 3) Process gen_slmy directory: filenames like "1_xxx.png"
    gen_paths = [
        p for p in glob.glob(os.path.join(gen_dir, "*"))
        if os.path.splitext(p)[1].lower() in exts
    ]
    gen_paths.sort()

    print(f"Found {len(gen_paths)} generated images in {gen_dir}")

    for gp in gen_paths:
        fname = os.path.basename(gp)  # e.g. "1_l1.20_s0.71_cfg9.5_steps70_seed3.png"
        m = id_pattern.match(fname)
        if not m:
            print(f"[WARN] skip generated image without leading id: {fname}")
            continue
        img_id = m.group(1)

        if img_id not in mask_index:
            print(f"[WARN] no mask for generated id={img_id}")
            continue

        mask_path = mask_index[img_id]

        try:
            img = load_gray_image(gp)
            mask = load_gray_image(mask_path)
            metrics = compute_metrics(img, mask, radius=radius)

            rows.append({
                "id": img_id,
                "group": "SLGP",
                "img_file": fname,
                "mask_file": os.path.basename(mask_path),
                "mu_g_in": metrics["mu_g_in"],
                "mu_g_out": metrics["mu_g_out"],
                "Delta_G": metrics["Delta_G"],
                "C_in": metrics["C_in"],
                "C_out": metrics["C_out"],
                "Delta_C": metrics["Delta_C"],
            })
        except Exception as e:
            print(f"[ERROR] generated id={img_id}, file={fname}: {e}")

    # 4) Write CSV
    fieldnames = [
        "id", "group", "img_file", "mask_file",
        "mu_g_in", "mu_g_out", "Delta_G",
        "C_in", "C_out", "Delta_C",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nDone. Total rows written: {len(rows)}")
    print(f"CSV saved to: {out_csv}")


if __name__ == "__main__":
    main()
