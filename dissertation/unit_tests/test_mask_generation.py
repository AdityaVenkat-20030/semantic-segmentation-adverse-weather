from pathlib import Path
from PIL import Image
import numpy as np

from generate_masks import create_mask

json_file = Path(
    "dissertation/data/idd_aw/IDDAW/val/FOG/gtSeg/69/00000000_mask.json"
)

mask = create_mask(json_file)

print("Shape:", mask.shape)
print("Unique values:", np.unique(mask))

Image.fromarray(mask).save("test_mask.png")

print("Saved test_mask.png")