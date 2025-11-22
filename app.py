import gradio as gr
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
from PIL import Image

# Force CPU
torch.cuda.is_available = lambda: False
device = torch.device("cpu")

# Global state
predictor = None
current_image = None
masks = []
points = []

def load_model():
    global predictor
    if predictor is None:
        print("Loading SAM2 model...")
        model_id = "facebook/sam2.1-hiera-tiny"
        predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
        print("Model loaded.")

def process_image(image):
    global current_image, masks, points
    if image is None:
        return None, "0"
    
    # Initialize/Reset for new image
    current_image = image
    masks = []
    points = []
    
    load_model()
    predictor.set_image(image)
    
    return image, "0"

def on_select(evt: gr.SelectData):
    global masks, points, current_image
    
    if current_image is None or predictor is None:
        return current_image, str(len(masks))
    
    x, y = evt.index[0], evt.index[1]
    print(f"Clicked at: {x}, {y}")
    
    # Predict
    with torch.inference_mode():
        out_masks, scores, _ = predictor.predict(
            point_coords=[[x, y]],
            point_labels=[1],
            multimask_output=False
        )
    
    mask = out_masks[0].astype("uint8")
    masks.append(mask)
    points.append((x, y))
    
    # Draw overlay
    # Create a green overlay
    overlay = current_image.copy()
    
    # We need to combine all masks
    combined_mask = np.zeros_like(masks[0])
    for m in masks:
        combined_mask = np.logical_or(combined_mask, m)
        
    # Apply green tint where mask is present
    # overlay is RGB (Gradio uses RGB numpy arrays)
    overlay[combined_mask > 0] = overlay[combined_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    
    return overlay, str(len(masks))

def undo():
    global masks, points, current_image
    if not masks:
        return current_image if current_image is not None else None, "0"
        
    masks.pop()
    points.pop()
    
    if current_image is None:
        return None, "0"
        
    overlay = current_image.copy()
    if masks:
        combined_mask = np.zeros_like(masks[0])
        for m in masks:
            combined_mask = np.logical_or(combined_mask, m)
        overlay[combined_mask > 0] = overlay[combined_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        
    return overlay, str(len(masks))

def reset():
    global masks, points, current_image
    masks = []
    points = []
    return current_image, "0"

with gr.Blocks(title="Compteur de Bûches SAM2") as demo:
    gr.Markdown("# 🪵 Compteur de Bûches avec SAM2")
    gr.Markdown("Chargez une image, puis cliquez sur les bûches pour les compter.")
    
    with gr.Row():
        with gr.Column(scale=3):
            img_input = gr.Image(label="Image", type="numpy", interactive=True)
        
        with gr.Column(scale=1):
            count_output = gr.Label(value="0", label="Nombre de bûches")
            btn_undo = gr.Button("↩️ Annuler dernier")
            btn_reset = gr.Button("🗑️ Réinitialiser")

    # Events
    img_input.upload(process_image, inputs=img_input, outputs=[img_input, count_output])
    img_input.select(on_select, outputs=[img_input, count_output])
    
    btn_undo.click(undo, outputs=[img_input, count_output])
    btn_reset.click(reset, outputs=[img_input, count_output])

if __name__ == "__main__":
    demo.launch()
