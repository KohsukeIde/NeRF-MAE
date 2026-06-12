#!/usr/bin/env python3
"""Download selected Front3D rendered NeRF views from the NeRF-RPN release.

The RPN/detection data in this repository only contains radiance-density grids
and box annotations. NeRF-MAE/NeRF-RPN-style visual figures require the rendered
RGB views and camera transforms, which are released separately as
`front3d_nerf_data.zip` on Hugging Face.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from remotezip import RemoteZip


URL = "https://huggingface.co/datasets/lyclyc52/NeRF_RPN/resolve/main/front3d_nerf_data.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", required=True, help="Scene name, e.g. 3dfront_0131_00")
    parser.add_argument(
        "--output-dir",
        default="figures_src/fig1_render_view_assets/nerf_rpn_front3d_nerf_data",
        help="Directory where front3d_nerf_data/<scene>/... will be extracted.",
    )
    parser.add_argument("--include-model", action="store_true", help="Also extract train/model.msgpack.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with RemoteZip(URL) as zip_file:
        names = zip_file.namelist()
        wanted: list[str] = []
        for scene in args.scene:
            prefix = f"front3d_nerf_data/{scene}/train/"
            wanted.extend(
                n
                for n in names
                if n.startswith(prefix + "images/") and n.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            wanted.append(prefix + "transforms.json")
            if args.include_model:
                wanted.append(prefix + "model.msgpack")
        wanted = [n for n in wanted if n in names]
        print(f"extracting {len(wanted)} files to {out}")
        for index, name in enumerate(wanted, start=1):
            zip_file.extract(name, out)
            if index % 100 == 0:
                print(f"  extracted {index}/{len(wanted)}")
    print("done")


if __name__ == "__main__":
    main()
