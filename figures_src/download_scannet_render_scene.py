#!/usr/bin/env python3
"""Download selected ScanNet rendered RGB views from the NeRF-RPN release."""

from __future__ import annotations

import argparse
from pathlib import Path

from remotezip import RemoteZip


URL = "https://huggingface.co/datasets/lyclyc52/NeRF_RPN/resolve/main/scannet_nerf_data.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", required=True, help="Scene name, e.g. scene0151_00")
    parser.add_argument(
        "--output-dir",
        default="figures_src/qualitative_detection_assets/render_cache/scannet_render_data",
        help="Directory where scannet_nerf_data/<scene>/... will be extracted.",
    )
    parser.add_argument("--include-test", action="store_true", help="Also extract test/rgb/*.jpg.")
    parser.add_argument("--include-depth", action="store_true", help="Also extract depth images.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with RemoteZip(URL) as zip_file:
        names = zip_file.namelist()
        wanted: list[str] = []
        for scene in args.scene:
            prefix = f"scannet_nerf_data/{scene}/"
            for filename in ["transforms_train.json", "transforms_test.json", "config.json"]:
                wanted.append(prefix + filename)
            wanted.extend(
                n
                for n in names
                if n.startswith(prefix + "train/rgb/") and n.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            if args.include_test:
                wanted.extend(
                    n
                    for n in names
                    if n.startswith(prefix + "test/rgb/") and n.lower().endswith((".jpg", ".jpeg", ".png"))
                )
            if args.include_depth:
                wanted.extend(
                    n
                    for n in names
                    if (n.startswith(prefix + "train/depth/") or n.startswith(prefix + "test/depth/"))
                    and n.lower().endswith((".jpg", ".jpeg", ".png"))
                )
        wanted = [n for n in wanted if n in names]
        print(f"extracting {len(wanted)} files to {out}")
        for index, name in enumerate(wanted, start=1):
            zip_file.extract(name, out)
            if index % 100 == 0:
                print(f"  extracted {index}/{len(wanted)}")
    print("done")


if __name__ == "__main__":
    main()
