#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch no-defect images + specified mask → diffusion-generate “steel-core fracture” defect samples
- Iterate over *_json subfolders under root_dir (Labelme-exported dataset structure); each folder contains img.png and label.png
- For each no-defect base image, sweep a LoRA/strength/seed grid and call SD Inpainting to generate candidate images
- Filter using a trained YOLO11 detection model: if any detection box has conf >= min_conf, it is deemed a “qualified defect sample”
- For each base image, keep at most per_image_target qualified samples; stop early once the target is reached and move to the next base image
- If the seed range is exhausted and still < per_image_target, also move to the next image
- Save all qualified samples to out_dir

Dependencies:
- diffusers >= 0.24
- ultralytics >= 8.x (YOLO11)
- pillow, numpy, torch
"""

import argparse, os, gc
from pathlib import Path
from typing import List
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from ultralytics import YOLO

# ============ Utility functions from your original script (kept and lightly wrapped) ============
def load_rgb(path):
    return Image.open(path).convert("RGB")

def binarize_labelme_mask(mask_im: Image.Image, target_size):
    if mask_im.mode != "L":
        mask_im = mask_im.convert("L")
    if mask_im.size != target_size:
        mask_im = mask_im.resize(target_size, Image.NEAREST)
    arr = np.array(mask_im)
    bin_arr = np.where(arr > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(bin_arr, mode="L")

def maybe_invert_mask(mask_im: Image.Image, force_invert=False):
    if force_invert:
        return Image.fromarray(255 - np.array(mask_im), mode="L"), True
    arr = np.array(mask_im, dtype=np.uint8)
    white_ratio = (arr > 127).mean()
    if white_ratio > 0.5:
        inv = Image.fromarray(255 - arr, mode="L")
        print(f"[mask] white_ratio={white_ratio:.2f} > 0.5, auto invert mask.")
        return inv, True
    return mask_im, False

def parse_float_list(v, fallback):
    """Comma-separated list of floats; if empty, use fallback (single-value list)"""
    if v is None or str(v).strip() == "":
        return [fallback]
    return [float(x) for x in str(v).split(",")]

def resolve_lora_weight(lora_dir_or_file: str):
    """
    Resolve LoRA path:
    - If a file path is provided (.safetensors/.bin), directly return (dir, filename)
    - If a directory is provided, automatically search for common filenames under the directory and return
    """
    p = Path(lora_dir_or_file)
    if p.is_file():
        return str(p.parent), p.name
    candidates = [
        "pytorch_lora_weights.safetensors",
        "adapter_model.safetensors",
        "pytorch_lora_weights.bin",
        "adapter_model.bin",
    ]
    for name in candidates:
        f = p / name
        if f.exists():
            return str(p), name
    raise FileNotFoundError(
        f"Cannot find LoRA weight file in {lora_dir_or_file}. "
        f"Tried: {', '.join(candidates)}. "
        f"Please set --lora_dir to the folder or the exact file path."
    )

# ==================== YOLO11 confidence filtering ====================
def yolo_pass(det_model: YOLO, pil_img: Image.Image, min_conf: float, accept_classes: List[int] = None) -> bool:
    """
    Perform object detection on the generated image:
    - If any detection box has conf >= min_conf (and class is within accept_classes if specified), deem it “qualified”
    """
    # Ultralytics YOLO supports passing PIL directly
    res_list = det_model.predict(pil_img, verbose=False)
    if not res_list:
        return False
    res = res_list[0]
    if res.boxes is None or len(res.boxes) == 0:
        return False
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    for c, k in zip(confs, clss):
        if c >= min_conf and (accept_classes is None or k in accept_classes):
            return True
    return False

# ==================== Main pipeline ====================
def build_pipe_and_adapter(args, device):
    base_model = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16",
        # For fully offline usage, you can disable the safety checker:
        # safety_checker=None,
        # feature_extractor=None,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    active_adapter = None
    if args.lora_dir and len(str(args.lora_dir).strip()) > 0:
        lora_root, weight_name = resolve_lora_weight(args.lora_dir)
        pipe.load_lora_weights(
            lora_root,
            adapter_name="fracture",
            weight_name=weight_name,
            local_files_only=True
        )
        active_adapter = "fracture"
        print(f"[lora] loaded from {lora_root} (weight_name={weight_name})")
    return pipe, active_adapter

def generate_and_filter_for_one_image(
    pipe: StableDiffusionInpaintPipeline,
    active_adapter: str,
    img: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative: str,
    device: str,
    lora_list: List[float],
    strength_list: List[float],
    seed_iter: List[int],
    steps: int,
    cfg: float,
    det_model: YOLO,
    min_conf: float,
    per_image_target: int,
    out_dir: Path,
    base_tag: str,
    accept_classes: List[int] = None,
):
    saved = 0
    # Save the mask used for this image once (save only for the first image to avoid duplicates)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    mask.save(out_dir / "masks" / f"{base_tag}_mask_used.png")

    for ls in lora_list:
        if active_adapter and hasattr(pipe, "set_adapters"):
            pipe.set_adapters([active_adapter], adapter_weights=[ls])
        for st in strength_list:
            for sd in seed_iter:
                if saved >= per_image_target:
                    print(f"[skip] {base_tag}: already reached {per_image_target} images, stopping early.")
                    return saved

                g = None if sd < 0 else torch.Generator(device=device).manual_seed(sd)

                extra_kwargs = {}
                # Optional arguments compatible with different diffusers versions
                if "guidance_rescale" in pipe.__call__.__code__.co_varnames:
                    extra_kwargs["guidance_rescale"] = 0.7
                if "noise_offset" in pipe.__call__.__code__.co_varnames:
                    extra_kwargs["noise_offset"] = 0.01

                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=img,
                    mask_image=mask,            # white=edit, black=keep original
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    strength=st,
                    generator=g,
                    **extra_kwargs
                ).images[0]

                # YOLO detection filtering
                ok = yolo_pass(det_model, image, min_conf=min_conf, accept_classes=accept_classes)
                if ok:
                    fname = f"{base_tag}_l{ls:.2f}_s{st:.2f}_cfg{cfg:.1f}_steps{steps}_seed{sd}.png"
                    out_path = out_dir / fname
                    image.save(out_path)
                    saved += 1
                    print(f"[SAVE✓] {out_path}  (saved={saved}/{per_image_target})")
                else:
                    print(f"[DROP] {base_tag} seed={sd} does not meet confidence threshold {min_conf}")

                # Release intermediate variables
                del image
                if device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
    return saved

def find_json_subfolders(root_dir: Path) -> List[Path]:
    # Only take subfolders named *_json (consistent with labelme_json_to_dataset output)
    return sorted([p for p in root_dir.iterdir() if p.is_dir() and p.name.endswith("_json")])

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse LoRA / strength / seed
    lora_list = parse_float_list(args.lora_list, args.lora_scale)
    strength_list = parse_float_list(args.strength_list, args.strength)
    if args.seed_start is not None and args.seed_end is not None:
        seed_iter = list(range(int(args.seed_start), int(args.seed_end) + 1))
    else:
        seed_iter = [int(args.seed)]

    # Prompts (customize/override via args as needed)
    prompt = (
        "x-ray radiograph of a single transmission splice sleeve with only one steel core, "
        "with exactly one steel-core fracture defect in the center, "
        "no multiple fractures, clear discontinuity"
    )
    negative = (
        "extra rods, multiple splice sleeves, top bar, bottom bar, "
        "duplicate steel wires, repeated structures, frame, border, "
        "background noise, artifacts"
    )

    # Initialize diffusion pipeline & LoRA
    pipe, active_adapter = build_pipe_and_adapter(args, device)

    # Load YOLO11 detection model (load once)
    det_model = YOLO(args.yolo_model)
    print(f"[yolo] loaded: {args.yolo_model}")

    # Optional class filtering (default None = accept any class; only check conf)
    accept_classes = None
    if args.accept_classes and args.accept_classes.strip():
        accept_classes = [int(x) for x in args.accept_classes.split(",")]
        print(f"[yolo] accept classes: {accept_classes}")

    # Iterate over *_json subfolders under root_dir
    root = Path(args.root_dir).expanduser().resolve()
    subdirs = find_json_subfolders(root)
    if not subdirs:
        print(f"[warn] No *_json subfolders found under {root}.")
        return
    print(f"[info] Found {len(subdirs)} *_json subfolders in total.")

    # Process each subfolder
    for d in subdirs:
        img_path = d / "img.png"
        mask_path = d / "label.png"
        if not img_path.exists() or not mask_path.exists():
            print(f"[skip] {d} missing img.png or label.png")
            continue

        # Read base image & mask (auto-binarize / invert if needed per your logic)
        img = load_rgb(str(img_path))
        raw_mask = Image.open(str(mask_path))
        mask = binarize_labelme_mask(raw_mask, img.size)
        mask, inverted = maybe_invert_mask(mask, force_invert=args.invert_mask)
        print(f"[mask] {d.name}: size={mask.size}, inverted={inverted}")

        base_tag = d.name.replace("_json", "")  # e.g., 132_json -> 132
        saved = generate_and_filter_for_one_image(
            pipe, active_adapter, img, mask,
            prompt, negative, device,
            lora_list, strength_list, seed_iter,
            steps=args.steps, cfg=args.cfg,
            det_model=det_model,
            min_conf=args.min_conf,
            per_image_target=args.per_image_target,
            out_dir=out_dir,
            base_tag=base_tag,
            accept_classes=accept_classes,
        )
        print(f"[done] {base_tag}: qualified samples {saved}/{args.per_image_target}")

    print(f"[ALL DONE] Qualified samples output directory: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # ====== Added: batch root directory & unified output directory ======
    ap.add_argument("--root_dir", type=str,
                    default="/home/zgc/datawork/no_defect_xray",
                    help="Contains multiple *_json subfolders (each contains img.png & label.png)")
    ap.add_argument("--out_dir", type=str, default="inpaint_selected_out",
                    help="Unified output directory for all qualified samples")

    # ====== LoRA & Inpainting inference parameters (following your original script logic) ======
    ap.add_argument("--lora_dir", default="outputs0912/lora-ssfracture/checkpoint-500",
                    help="LoRA directory or direct file path (*.safetensors / *.bin)")
    ap.add_argument("--steps", type=int, default=70)
    ap.add_argument("--cfg", type=float, default=9.5)
    ap.add_argument("--invert_mask", action="store_true", help="Force invert mask (white↔black)")

    # Fixed seed (when not sweeping)
    ap.add_argument("--seed", type=int, default=2025, help="Single fixed seed (if <0 means random)")

    # Seed range (inclusive); if both are provided, enable sweeping
    ap.add_argument("--seed_start", type=int, default=1, help="Start seed")
    ap.add_argument("--seed_end",   type=int, default=1500, help="End seed (inclusive)")

    # Single values
    ap.add_argument("--lora_scale", type=float, default=0.95, help="LoRA scale for a single inference")
    ap.add_argument("--strength",   type=float, default=0.60, help="Strength for a single inference (recommended 0.4~0.7)")

    # List sweep (comma-separated; if empty, use the single values above)
    ap.add_argument("--lora_list",     type=str, default="1.2",
                    help="LoRA scale list (comma-separated)")
    ap.add_argument("--strength_list", type=str, default="0.71",
                    help="Strength list (comma-separated)")

    # ====== YOLO11 detection filtering parameters ======
    ap.add_argument("--yolo_model", type=str,
                    default="/home/zgc/datawork/DRimage/ultralytics-main0924/ultralytics-main/runs/detect/train3/weights/best.pt",
                    help="Trained YOLO11 detection model path")
    ap.add_argument("--min_conf", type=float, default=0.65,
                    help="Minimum confidence threshold for qualified samples")
    ap.add_argument("--accept_classes", type=str, default="",
                    help="Optional: restrict accepted class ID list (comma-separated). Empty = accept any class.")

    # ====== Per-image target number of qualified samples ======
    ap.add_argument("--per_image_target", type=int, default=10,
                    help="Maximum number of qualified defect samples to keep per no-defect base image")

    args = ap.parse_args()
    main(args)
