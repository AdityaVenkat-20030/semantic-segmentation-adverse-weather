#!/usr/bin/env python3

"""
Generate semantic segmentation masks from IDD / IDD-AW polygon annotations.

Input:    *_polygons.json

Output:
    Indexed PNG masks

"""

import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from dissertation.configs.label_mapping import (
    LABEL_MAP,
    IGNORE_INDEX,
    NUM_CLASSES,
)


# ============================================================
# POLYGON → MASK
# ============================================================

def create_mask(json_file: Path):

    with open(json_file, "r") as f:
        annotation = json.load(f)

    width = annotation["imgWidth"]
    height = annotation["imgHeight"]

    mask = Image.new("L", (width, height), IGNORE_INDEX)

    draw = ImageDraw.Draw(mask)

    objects = annotation.get("objects", [])

    for obj in objects:

        label = obj.get("label", "").strip()

        if label not in LABEL_MAP:
            continue

        class_id = LABEL_MAP[label]

        polygon = obj.get("polygon", [])

        if len(polygon) < 3:
            continue

        polygon = [tuple(point) for point in polygon]

        draw.polygon(
            polygon,
            fill=class_id
        )

    return np.array(mask, dtype=np.uint8)


# ============================================================
# MAIN
# ============================================================

def process_dataset(input_root, output_root):

    input_root = Path(input_root)
    output_root = Path(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    json_files = sorted(list(input_root.rglob("*.json")))

    print(f"Found {len(json_files)} annotation files")

    for json_file in tqdm(json_files):

        relative_path = json_file.relative_to(input_root)

        output_file = (
            output_root /
            relative_path.with_suffix(".png")
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        try:

            mask = create_mask(json_file)

            Image.fromarray(mask).save(output_file)

        except Exception as e:

            print(
                f"\nERROR: {json_file}\n{e}"
            )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_root",
        required=True
    )

    parser.add_argument(
        "--output_root",
        required=True
    )

    args = parser.parse_args()

    process_dataset(
        args.input_root,
        args.output_root
    )

    print("\nMask generation complete")


if __name__ == "__main__":
    main()