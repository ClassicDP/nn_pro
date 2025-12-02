#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import cv2
import onnxruntime as ort

def order_corners(pts):
    """Sort corners: [top-left, top-right, bottom-right, bottom-left]."""
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32).flatten()

class LPCornerDetectorRpi:
    def __init__(self, model_path):
        # Initialize ONNX Runtime
        # On RPi try to use standard CPU provider (or ACL if available)
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (224, 224)
        
        # Precompute coordinate grids for CoordConv
        h, w = self.input_shape
        self.xx_channel = np.tile(np.linspace(0, 1, w), (h, 1)).astype(np.float32)
        self.yy_channel = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w)).astype(np.float32)
        
        # ImageNet stats
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, img_bgr):
        """Prepare image: Resize -> Norm -> CoordConv."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.input_shape)
        
        # Normalize
        img_norm = img_resized.astype(np.float32) / 255.0
        img_norm = (img_norm - self.mean) / self.std
        
        # CHW
        img_chw = img_norm.transpose(2, 0, 1)
        
        # Add Coord Channels (3 -> 5)
        input_data = np.concatenate([
            img_chw,
            self.xx_channel[np.newaxis, ...],
            self.yy_channel[np.newaxis, ...]
        ], axis=0)
        
        # Add Batch Dim
        return input_data[np.newaxis, ...]

    def detect(self, img_path, output_dir):
        fname = os.path.basename(img_path)
        print(f"\nProcessing {fname}...")
        
        # 1. Load Image
        t0 = time.time()
        img_raw = cv2.imread(img_path)
        if img_raw is None:
            print(f"Error reading {img_path}")
            return
        h_orig, w_orig = img_raw.shape[:2]
        t_load = time.time() - t0
        
        # 2. Stage 1: Full Frame
        t1 = time.time()
        input_tensor = self.preprocess(img_raw)
        t_prep1 = time.time() - t1
        
        t2 = time.time()
        pred_s1 = self.session.run(None, {self.input_name: input_tensor})[0][0]
        t_infer1 = time.time() - t2
        
        # 3. Logic: Crop & Zoom
        t3 = time.time()
        pts_s1 = pred_s1.reshape(4, 2)
        
        # Bbox
        x_min, y_min = pts_s1[:, 0].min(), pts_s1[:, 1].min()
        x_max, y_max = pts_s1[:, 0].max(), pts_s1[:, 1].max()
        
        w_box = x_max - x_min
        h_box = y_max - y_min
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        
        # Padding 1.5x (Safe Crop)
        crop_size_x = max(w_box * 2.0, 0.2)
        crop_size_y = max(h_box * 2.0, 0.2)
        
        x1 = np.clip(cx - crop_size_x / 2, 0, 1)
        x2 = np.clip(cx + crop_size_x / 2, 0, 1)
        y1 = np.clip(cy - crop_size_y / 2, 0, 1)
        y2 = np.clip(cy + crop_size_y / 2, 0, 1)
        
        x1_px, x2_px = int(x1 * w_orig), int(x2 * w_orig)
        y1_px, y2_px = int(y1 * h_orig), int(y2 * h_orig)
        
        # Check crop validity
        if x2_px - x1_px < 16 or y2_px - y1_px < 16:
            print("Warning: Stage 1 crop too small, skipping Stage 2")
            pts_final = pts_s1
            valid_stage2 = False
        else:
            img_crop = img_raw[y1_px:y2_px, x1_px:x2_px]
            valid_stage2 = True
            
        t_crop = time.time() - t3
        
        # 4. Stage 2: Zoomed
        t_prep2 = 0
        t_infer2 = 0
        t_remap = 0
        
        if valid_stage2:
            t4 = time.time()
            input_crop = self.preprocess(img_crop)
            t_prep2 = time.time() - t4
            
            t5 = time.time()
            pred_s2 = self.session.run(None, {self.input_name: input_crop})[0][0]
            t_infer2 = time.time() - t5
            
            t6 = time.time()
            # Remap to original coords
            pts_s2 = pred_s2.reshape(4, 2)
            crop_w_norm = x2 - x1
            crop_h_norm = y2 - y1
            
            pts_final = np.zeros_like(pts_s2)
            pts_final[:, 0] = x1 + pts_s2[:, 0] * crop_w_norm
            pts_final[:, 1] = y1 + pts_s2[:, 1] * crop_h_norm
            t_remap = time.time() - t6
            
        # 5. Visualization
        t7 = time.time()
        vis = img_raw.copy()
        
        # Stage 1 (Blue)
        s1_px = pts_s1.copy()
        s1_px[:, 0] *= w_orig
        s1_px[:, 1] *= h_orig
        s1_px = order_corners(s1_px)
        cv2.polylines(vis, [s1_px.astype(np.int32).reshape(-1, 2)], True, (255, 0, 0), 2)
        
        # Stage 2 (Red)
        s2_px = pts_final.copy()
        s2_px[:, 0] *= w_orig
        s2_px[:, 1] *= h_orig
        s2_px = order_corners(s2_px)
        cv2.polylines(vis, [s2_px.astype(np.int32).reshape(-1, 2)], True, (0, 0, 255), 2)
        
        # Stats
        total_time = (time.time() - t0) * 1000
        net_time = (t_infer1 + t_infer2) * 1000
        
        cv2.putText(vis, f"Total: {total_time:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Net: {net_time:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, vis)
        t_vis = time.time() - t7
        
        print(f"  [Timing]")
        print(f"  Load:    {t_load*1000:.1f} ms")
        print(f"  S1 Prep: {t_prep1*1000:.1f} ms")
        print(f"  S1 Inf:  {t_infer1*1000:.1f} ms")
        print(f"  Crop:    {t_crop*1000:.1f} ms")
        if valid_stage2:
            print(f"  S2 Prep: {t_prep2*1000:.1f} ms")
            print(f"  S2 Inf:  {t_infer2*1000:.1f} ms")
            print(f"  Remap:   {t_remap*1000:.1f} ms")
        print(f"  Vis/Save:{t_vis*1000:.1f} ms")
        print(f"  ---------------------")
        print(f"  TOTAL:   {total_time:.1f} ms")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--model', default='model/lp_regressor.onnx', help='Path to ONNX model')
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    detector = LPCornerDetectorRpi(args.model)
    
    # Warmup
    print("Warming up...")
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    detector.preprocess(dummy)
    
    files = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for f in files:
        detector.detect(os.path.join(args.input, f), args.output)

if __name__ == '__main__':
    main()

