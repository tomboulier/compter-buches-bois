import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os

# Force CPU for simplicity/compatibility as per original script
torch.cuda.is_available = lambda: False
device = torch.device("cpu")

class LogCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Compteur de Bûches - SAM2")
        self.root.geometry("1200x800")

        self.model_id = "facebook/sam2.1-hiera-tiny"
        self.predictor = None
        self.image_path = "buches.jpg" # Default
        self.image = None
        self.original_image_bgr = None
        self.masks = []
        self.points = []
        self.tk_image = None
        
        # Layout
        self.controls_frame = tk.Frame(root)
        self.controls_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        self.btn_load = tk.Button(self.controls_frame, text="Charger Image", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        self.btn_undo = tk.Button(self.controls_frame, text="Annuler dernier", command=self.undo_last)
        self.btn_undo.pack(side=tk.LEFT, padx=5)
        
        self.btn_reset = tk.Button(self.controls_frame, text="Réinitialiser", command=self.reset)
        self.btn_reset.pack(side=tk.LEFT, padx=5)
        
        self.lbl_count = tk.Label(self.controls_frame, text="Nombre de bûches: 0", font=("Arial", 16, "bold"))
        self.lbl_count.pack(side=tk.RIGHT, padx=20)

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)

        # Initialize Model
        self.load_model()
        
        # Load default image if exists
        if os.path.exists(self.image_path):
            self.process_image(self.image_path)

    def load_model(self):
        try:
            print("Loading SAM2 model...")
            self.predictor = SAM2ImagePredictor.from_pretrained(self.model_id, device=device)
            print("Model loaded.")
        except Exception as e:
            messagebox.showerror("Erreur Modèle", f"Impossible de charger SAM2:\n{e}")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if path:
            self.image_path = path
            self.process_image(path)

    def process_image(self, path):
        self.original_image_bgr = cv2.imread(path)
        if self.original_image_bgr is None:
            return
        
        image_rgb = cv2.cvtColor(self.original_image_bgr, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(image_rgb)
        
        # Set image for SAM2
        if self.predictor:
            self.predictor.set_image(image_rgb)
        
        self.reset_data()
        self.display_image()

    def reset_data(self):
        self.masks = []
        self.points = []
        self.update_count()

    def display_image(self):
        if self.image is None:
            return
            
        # Resize for display if too large
        display_w, display_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if display_w < 10 or display_h < 10:
            display_w, display_h = 1000, 700 # Default fallback
            
        img_w, img_h = self.image.size
        scale = min(display_w/img_w, display_h/img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        
        resized_image = self.image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Overlay masks
        if self.masks:
            # Create an overlay on the resized image
            # This is a bit complex to do efficiently in PIL for many masks, 
            # but for a simple counter, we can draw on top.
            # Actually, better to draw on the base numpy array and convert.
            
            # Re-construct visualization from base
            vis_img = self.original_image_bgr.copy()
            for mask in self.masks:
                # mask is binary 2D
                vis_img[mask > 0] = (0, 255, 0) # Green paint
            
            vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            pil_vis = Image.fromarray(vis_img_rgb)
            resized_image = pil_vis.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.canvas.delete("all")
        
        # Center image
        x_offset = (display_w - new_w) // 2
        y_offset = (display_h - new_h) // 2
        
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.tk_image)
        self.scale = scale
        self.offset = (x_offset, y_offset)

    def on_click(self, event):
        if not self.predictor or not self.image:
            return
            
        # Map click to original image coordinates
        x_disp, y_disp = event.x, event.y
        x_off, y_off = self.offset
        
        x_orig = int((x_disp - x_off) / self.scale)
        y_orig = int((y_disp - y_off) / self.scale)
        
        # Check bounds
        w, h = self.image.size
        if 0 <= x_orig < w and 0 <= y_orig < h:
            self.predict_mask(x_orig, y_orig)

    def predict_mask(self, x, y):
        # Predict
        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=[[x, y]],
                point_labels=[1],
                multimask_output=False
            )
        
        mask = masks[0].astype("uint8")
        self.masks.append(mask)
        self.points.append((x, y))
        
        self.update_count()
        self.display_image()

    def undo_last(self):
        if self.masks:
            self.masks.pop()
            self.points.pop()
            self.update_count()
            self.display_image()

    def reset(self):
        self.reset_data()
        self.display_image()

    def update_count(self):
        self.lbl_count.config(text=f"Nombre de bûches: {len(self.masks)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LogCounterApp(root)
    root.mainloop()
