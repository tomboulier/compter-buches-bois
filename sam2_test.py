import cv2
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---- force CPU ----
torch.cuda.is_available = lambda: False
device = torch.device("cpu")

model_id = "facebook/sam2.1-hiera-tiny"
predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)

image_path = "buches.jpg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

predictor.set_image(image_rgb)

input_point = [600, 600]
input_label = 1

with torch.inference_mode():
    masks, scores, _ = predictor.predict(
        point_coords=[input_point],
        point_labels=[input_label],
        multimask_output=False
    )

mask = masks[0].astype("uint8") * 255
cv2.imwrite("mask_buche.png", mask)

overlay = image_bgr.copy()
overlay[mask > 0] = (0,255,0)
cv2.imwrite("segmented_buche.png", overlay)

print("Done.")
