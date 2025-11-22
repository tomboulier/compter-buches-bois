import cv2
import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---- SAM2 CPU ----
torch.cuda.is_available = lambda: False
device = torch.device("cpu")

model_id = "facebook/sam2.1-hiera-tiny"
predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)

# ---- Load image ----
image = cv2.imread("buches.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(rgb)

# ---- WATERSHED PREPROCESSING ----

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# Binary threshold
_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Morphology to clean noise
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)

# Background
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Distance transform (to find centers)
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Markers
num_labels, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Apply Watershed
markers = cv2.watershed(image, markers)

# Extract each object
overlay = image.copy()
count = 0

unique_markers = np.unique(markers)
for m in unique_markers:
    if m <= 1:
        continue

    mask_region = np.where(markers == m, 1, 0).astype(np.uint8)
    ys, xs = np.where(mask_region == 1)
    if len(xs) == 0:
        continue

    # Pick a central point as SAM2 prompt
    cx = int(xs.mean())
    cy = int(ys.mean())

    # SAM2 refine segmentation
    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=[[cx, cy]],
            point_labels=[1],
            multimask_output=False
        )

    mask = masks[0]
    overlay[mask > 0] = (0, 255, 0)
    count += 1

cv2.imwrite("all_buches_segmented.png", overlay)
print("Nombre de bûches segmentées :", count)
