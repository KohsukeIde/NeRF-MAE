#!/usr/bin/env python3
"""Static audit for NeRF-MAE encoder mask path.

The goal is to create an explicit report before implementing MixNeRF-MAE:
- does the model construct a masked full grid M?
- does the encoder process all H/p x W/p x D/p tokens?
- where are mask/zero tokens introduced?
- does loss use a separate removed-mask?

Run from repo root or `nerf_mae/`:
    python probe_scripts/encoder_mask_path_report.py --repo .. --out ../results/.../encoder_mask_path_report.md
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PATTERNS = {
    "forward_loss": r"def\s+forward_loss",
    "forward": r"def\s+forward\(",
    "forward_encoder_decoder": r"forward_encoder_e?coder",
    "masking_prob": r"masking_prob|mask_ratio|masking_ratio|mask_prob",
    "mask_remove": r"mask_remove|removed_mask|mask_patches",
    "occupied_rgb": r"target_alpha\s*>\s*0\.01|probe_rgb_loss.*occupied|RGB_LOSS_MODES",
    "swin_stages": r"Swin|Transformer\s*Stage|patch_partition|patch_partitioning|patch merging",
    "pos_embed": r"pos_embed|positional",
}

FILES = [
    "nerf_mae/model/mae/swin_mae3d.py",
    "nerf_mae/model/mae/shortcut_probe.py",
    "nerf_mae/run_swin_mae3d.py",
]


def snippets(text: str, pattern: str, window: int = 160):
    out = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        s = max(0, m.start() - window)
        e = min(len(text), m.end() + window)
        out.append(text[s:e].replace("\n", " "))
        if len(out) >= 5:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(".."))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    lines = []
    lines.append("# Encoder mask-path report")
    lines.append("")
    lines.append(f"Repo: `{args.repo.resolve()}`")
    lines.append("")
    lines.append("## Pattern summary")
    lines.append("")
    lines.append("| file | pattern | count | sample |")
    lines.append("|---|---|---:|---|")

    for rel in FILES:
        path = args.repo / rel
        if not path.exists():
            lines.append(f"| `{rel}` | _missing_ | 0 | file not found |")
            continue
        text = path.read_text(errors="replace")
        for name, pat in PATTERNS.items():
            matches = list(re.finditer(pat, text, flags=re.IGNORECASE))
            sample = snippets(text, pat, window=80)
            sample_txt = sample[0].replace("|", "\\|") if sample else ""
            lines.append(f"| `{rel}` | `{name}` | {len(matches)} | `{sample_txt[:240]}` |")

    lines.append("")
    lines.append("## Checklist")
    lines.append("")
    lines.append("- [ ] Confirm whether `forward_encoder_ecoder` receives a full dense grid after masking.")
    lines.append("- [ ] Confirm whether masked positions are zeros, learned mask tokens, or otherwise marked.")
    lines.append("- [ ] Confirm whether the removed mask is used only in the loss or also in the encoder.")
    lines.append("- [ ] Confirm whether patch-grid shape is preserved through all Swin stages.")
    lines.append("- [ ] Confirm whether disabling base internal masking sets base returned mask mean to 0 during MixNeRF runs.")

    text = "\n".join(lines) + "\n"
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
