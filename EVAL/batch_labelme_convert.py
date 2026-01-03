#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-call labelme_json_to_dataset to convert all *.json files under a specified directory into datasets
"""

import os
import glob
import subprocess

def batch_convert(json_dir):
    # Ensure the directory exists
    if not os.path.exists(json_dir):
        print(f"[Error] Path does not exist: {json_dir}")
        return

    # Find all json files
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    if not json_files:
        print(f"[Info] No .json files found in {json_dir}")
        return

    print(f"[Info] Found {len(json_files)} JSON files in total, starting conversion...")

    for json_file in json_files:
        print(f"[Processing] {json_file}")
        try:
            # Call the built-in labelme script
            subprocess.run(
                ["labelme_json_to_dataset", json_file],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[Failed] Conversion failed for {json_file}: {e}")

    print("[Done] All conversions finished!")

if __name__ == "__main__":
    # Modify to the directory where your JSON files are located
    json_dir = "/home/zgc/datawork/no_defect_xray"
    batch_convert(json_dir)
