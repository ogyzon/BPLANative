# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from lightglue import SuperPoint, LightGlue
from lightglue.utils import rbd
import threading


class CustomVideoPlayer:
    """
    Custom video player that can draw overlays (keypoints, lines) on each frame
    before displaying it to the user.
    """
    
    def __init__(self, parent, app, width=640, height=480, on_frame_callback=None):
        self.parent = parent
        self.parent_app = app  # Reference to main App
        self.width = width
        self.height = height
        self.on_frame_callback = on_frame_callback
        
        # Main frame for video player
        self.frame = tk.Frame(parent, bg='black')
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for video display
        self.canvas = tk.Canvas(self.frame, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons frame
        self.control_frame = tk.Frame(self.frame)
        self.control_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.btn_play = tk.Button(self.control_frame, text="Play", command=self.play)
        self.btn_play.pack(side=tk.LEFT, padx=2)
        
        self.btn_pause = tk.Button(self.control_frame, text="Pause", command=self.pause)
        self.btn_pause.pack(side=tk.LEFT, padx=2)
        
        self.btn_stop = tk.Button(self.control_frame, text="Stop", command=self.stop)
        self.btn_stop.pack(side=tk.LEFT, padx=2)
        
        self.seek_scale = tk.Scale(self.control_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.seek)
        self.seek_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.time_label = tk.Label(self.control_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.RIGHT, padx=5)
        
        # Video state
        self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.fps = 30
        self.current_frame = 0
        self.is_playing = False
        self.play_thread = None
        self.stop_thread = False
        self.current_photo = None
        
        # Overlay data (points to draw)
        self.match_points = []              # Points on video frame
        self.satellite_match_points = []    # Corresponding points on satellite
        self.homography_matrix = None       # Transformation matrix for drawing lines
        
        self.update_display()
    
    def load(self, video_path):
        """Load video file"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print("Cannot open video")
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30
        
        self.current_frame = 0
        self.show_frame(0)
        
        print(f"Video loaded: {self.total_frames} frames, {self.fps:.1f} FPS")
        return True
    
    def draw_combined_with_lines(self, video_frame, sat_image, video_points, sat_points, H=None, max_lines=20):
        """
        Create a combined image with satellite on left, video on right,
        and draw lines between matching points.
        If H (homography) is provided, also draw projected bounding box.
        """
        # Convert satellite PIL image to numpy array
        sat_np = np.array(sat_image.convert('RGB'))
        sat_np = cv2.cvtColor(sat_np, cv2.COLOR_RGB2BGR)
        
        # Resize satellite to match video frame height proportionally
        sat_h, sat_w = sat_np.shape[:2]
        video_h, video_w = video_frame.shape[:2]
        
        # Scale satellite to same height as video
        scale = video_h / sat_h
        new_sat_w = int(sat_w * scale)
        sat_resized = cv2.resize(sat_np, (new_sat_w, video_h))
        
        # Create combined image (satellite left, video right)
        combined = np.zeros((video_h, new_sat_w + video_w, 3), dtype=np.uint8)
        combined[:, :new_sat_w] = sat_resized
        combined[:, new_sat_w:] = video_frame
        
        # Scale satellite points to match resized satellite image
        if len(sat_points) > 0:
            sat_points_scaled = sat_points * [scale, scale]
        else:
            sat_points_scaled = []
        
        # Number of lines to draw
        num_lines = min(len(video_points), max_lines)
        
        for i in range(num_lines):
            if i < len(sat_points_scaled) and i < len(video_points):
                x1, y1 = sat_points_scaled[i]
                x2, y2 = video_points[i]
                
                # Draw green circle on satellite
                cv2.circle(combined, (int(x1), int(y1)), 5, (0, 255, 0), -1)
                
                # Draw red circle on video (offset by satellite width)
                cv2.circle(combined, (int(x2 + new_sat_w), int(y2)), 5, (0, 0, 255), -1)
                
                # Draw blue line between them
                cv2.line(combined, (int(x1), int(y1)), (int(x2 + new_sat_w), int(y2)), (255, 0, 0), 2)
        
        # If we have homography, draw projected satellite corners on video
        if H is not None and sat_points_scaled is not None and len(sat_points_scaled) >= 4:
            # Draw text showing RANSAC worked
            cv2.putText(combined, "RANSAC: OK", (new_sat_w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return combined
    
    def show_frame(self, frame_num):
        """Display a specific frame with overlay and lines"""
        if self.cap is None:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        
        if not ret:
            return
        
        # If we have matches, draw combined image with lines
        if len(self.match_points) >= 4 and len(self.satellite_match_points) >= 4 and self.parent_app.satellite_img is not None:
            combined = self.draw_combined_with_lines(
                frame, 
                self.parent_app.satellite_img,
                self.match_points, 
                self.satellite_match_points,
                self.homography_matrix,
                max_lines=20
            )
            display_frame = combined
        else:
            # Just show video frame
            display_frame = frame
            # Still draw points on video if any
            for (x, y) in self.match_points:
                cv2.circle(display_frame, (int(x), int(y)), 4, (0, 255, 0), -1)
        
        # Convert for display
        display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Get canvas size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w > 1 and canvas_h > 1:
            h, w = display_rgb.shape[:2]
            scale = min(canvas_w / w, canvas_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            display_rgb = cv2.resize(display_rgb, (new_w, new_h))
        else:
            new_w, new_h = display_rgb.shape[1], display_rgb.shape[0]
        
        # Display
        img = Image.fromarray(display_rgb)
        self.current_photo = ImageTk.PhotoImage(img)
        
        self.canvas.delete("all")
        x = (canvas_w - new_w) // 2 if canvas_w > 1 else 0
        y = (canvas_h - new_h) // 2 if canvas_h > 1 else 0
        self.canvas.create_image(x, y, image=self.current_photo, anchor=tk.NW)
        
        # Update time label
        current_time = frame_num / self.fps if self.fps > 0 else 0
        total_time = self.total_frames / self.fps if self.fps > 0 else 0
        self.time_label.config(text=f"{self._format_time(current_time)} / {self._format_time(total_time)}")
    
    def _format_time(self, seconds):
        """Format seconds to MM:SS"""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"
    
    def update_display(self):
        """Update display periodically (playback loop)"""
        if self.is_playing and self.cap is not None:
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                self.current_frame = 0
            
            self.show_frame(self.current_frame)
            if self.total_frames > 0:
                self.seek_scale.set(int(self.current_frame / self.total_frames * 100))
            
            # Call callback for frame processing
            if self.on_frame_callback and self.current_frame % 5 == 0:
                self.on_frame_callback(self.current_frame)
        
        self.parent.after(33, self.update_display)
    
    def set_match_points(self, video_points, satellite_points, H=None):
        """Set points to draw (called from matching thread)"""
        self.match_points = video_points
        self.satellite_match_points = satellite_points
        self.homography_matrix = H
        # Redraw current frame with new points
        if self.cap is not None:
            self.show_frame(self.current_frame)
    
    def play(self):
        """Start playback"""
        if self.cap is not None and not self.is_playing:
            self.is_playing = True
    
    def pause(self):
        """Pause playback"""
        self.is_playing = False
    
    def stop(self):
        """Stop and reset to beginning"""
        self.is_playing = False
        self.current_frame = 0
        self.show_frame(0)
        self.seek_scale.set(0)
    
    def seek(self, value):
        """Seek to position"""
        if self.cap is None or self.total_frames <= 0:
            return
        self.current_frame = int(int(value) / 100 * self.total_frames)
        self.show_frame(self.current_frame)
    
    def close(self):
        """Release resources"""
        self.is_playing = False
        if self.cap:
            self.cap.release()


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BPLA - Satellite to Drone Matching")
        self.root.geometry("1200x700")
        
        # Configure grid weights for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Left panel - Satellite photo
        left_frame = tk.LabelFrame(self.root, text="Satellite Image", font=("Arial", 10, "bold"))
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        
        self.photo_canvas = tk.Canvas(left_frame, bg='gray', highlightthickness=0)
        self.photo_canvas.grid(row=0, column=0, sticky="nsew")
        
        # Right panel - Video player
        right_frame = tk.LabelFrame(self.root, text="Drone Video", font=("Arial", 10, "bold"))
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        self.video_player = CustomVideoPlayer(right_frame, self, width=640, height=480, on_frame_callback=self.process_frame_background)
        
        # Bottom panel - Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10)
        
        self.btn_load_photo = tk.Button(btn_frame, text="Load Satellite Photo", command=self.load_photo, width=20)
        self.btn_load_photo.pack(side=tk.LEFT, padx=10)
        
        self.btn_load_video = tk.Button(btn_frame, text="Load Video", command=self.load_video, width=15)
        self.btn_load_video.pack(side=tk.LEFT, padx=10)
        
        # RANSAC settings
        self.ransac_threshold = 3.0
        tk.Label(btn_frame, text="RANSAC threshold:").pack(side=tk.LEFT, padx=(20,5))
        self.ransac_slider = tk.Scale(btn_frame, from_=1.0, to=10.0, orient=tk.HORIZONTAL, 
                                       resolution=0.5, length=150, command=self.update_ransac_threshold)
        self.ransac_slider.set(3.0)
        self.ransac_slider.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(btn_frame, text="Ready", font=("Arial", 9))
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # AI Models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = None
        self.matcher = None
        self.satellite_feats = None
        self.satellite_img = None
        self.satellite_keypoints = None
        self.satellite_display_photo = None
        
        # Matching thread
        self.matching_thread = None
        self.last_frame_processed = -1
        
        self.load_ai()
    
    def update_ransac_threshold(self, value):
        """Update RANSAC threshold from slider"""
        self.ransac_threshold = float(value)
        self.status_label.config(text=f"RANSAC threshold: {self.ransac_threshold}")
    
    def load_ai(self):
        """Load neural networks"""
        self.status_label.config(text="Loading AI...")
        self.root.update()
        
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        
        self.status_label.config(text=f"AI ready on {self.device}")
        print(f"AI loaded on {self.device}")
    
    def load_photo(self):
        """Load satellite photo"""
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if not path:
            return
        
        self.satellite_img = Image.open(path)
        print(f"Loaded photo: {path}, size: {self.satellite_img.size}")
        
        self.status_label.config(text="Extracting satellite features...")
        self.root.update()
        
        self.satellite_feats = self.extract_features(self.satellite_img)
        self.satellite_keypoints = self.extract_keypoints(self.satellite_img)
        
        self.display_photo_with_keypoints(self.satellite_keypoints)
        
        self.status_label.config(text=f"Satellite ready: {len(self.satellite_keypoints)} keypoints")
        print(f"Found {len(self.satellite_keypoints)} keypoints")
    
    def extract_features(self, img):
        """Extract full features for LightGlue matching"""
        img_np = np.array(img.convert('RGB'))
        tensor = transforms.ToTensor()(img_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.extractor.extract(tensor)
        return feats
    
    def extract_keypoints(self, img):
        """Extract only keypoint coordinates"""
        img_np = np.array(img.convert('RGB'))
        tensor = transforms.ToTensor()(img_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.extractor.extract(tensor)
        feats = rbd(feats)
        return feats['keypoints'].cpu().numpy()
    
    def display_photo_with_keypoints(self, keypoints, max_points=100):
        """Display satellite photo with green keypoints"""
        canvas_w = self.photo_canvas.winfo_width()
        canvas_h = self.photo_canvas.winfo_height()
        
        if canvas_w <= 1:
            canvas_w = 500
            canvas_h = 500
        
        img_copy = self.satellite_img.copy()
        from PIL import ImageDraw
        
        draw = ImageDraw.Draw(img_copy)
        for (x, y) in keypoints[:max_points]:
            draw.ellipse([x-3, y-3, x+3, y+3], fill='lime', outline='lime')
        
        img_copy.thumbnail((canvas_w, canvas_h))
        self.satellite_display_photo = ImageTk.PhotoImage(img_copy)
        
        self.photo_canvas.delete("all")
        x = (canvas_w - img_copy.width) // 2
        y = (canvas_h - img_copy.height) // 2
        self.photo_canvas.create_image(x, y, image=self.satellite_display_photo, anchor=tk.NW)
    
    def load_video(self):
        """Load video into player"""
        path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm")])
        if not path:
            return
        
        self.video_player.load(path)
        self.status_label.config(text="Video loaded. Click Play to start")
    
    def process_frame_background(self, frame_num):
        """Called from video player for each frame - starts background processing"""
        if self.satellite_feats is None:
            return
        
        if frame_num == self.last_frame_processed:
            return
        
        self.last_frame_processed = frame_num
        
        if self.matching_thread is None or not self.matching_thread.is_alive():
            self.matching_thread = threading.Thread(
                target=self.match_frame_in_background,
                args=(frame_num,),
                daemon=True
            )
            self.matching_thread.start()
    
    def ransac_filter(self, pts_sat, pts_frame, threshold=3.0, max_iter=2000):
        """
        Filter incorrect matches using RANSAC.
        
        Parameters:
        - pts_sat: points on satellite image (N, 2)
        - pts_frame: corresponding points on video frame (N, 2)
        - threshold: maximum allowed reprojection error (pixels)
        - max_iter: maximum number of RANSAC iterations
        
        Returns:
        - filtered_pts_sat: geometrically consistent points on satellite
        - filtered_pts_frame: corresponding points on video
        - H: homography matrix (transformation from satellite to frame)
        """
        if len(pts_sat) < 4:
            return pts_sat, pts_frame, None
        
        # Convert to proper format for OpenCV
        src_pts = pts_sat.reshape(-1, 1, 2).astype(np.float32)
        dst_pts = pts_frame.reshape(-1, 1, 2).astype(np.float32)
        
        # Find homography with RANSAC
        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=threshold,
            maxIters=max_iter,
            confidence=0.995
        )
        
        if H is not None and mask is not None:
            # Convert mask to boolean array
            mask = mask.ravel().astype(bool)
            filtered_pts_sat = pts_sat[mask]
            filtered_pts_frame = pts_frame[mask]
            return filtered_pts_sat, filtered_pts_frame, H
        else:
            return pts_sat, pts_frame, None
    
    def match_frame_in_background(self, frame_num):
        """Heavy matching work in background thread with RANSAC"""
        cap = cv2.VideoCapture(self.video_player.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return
        
        # Get matches from LightGlue
        video_points_raw, sat_points_raw = self.match_frame(frame)
        
        if len(video_points_raw) < 4:
            self.root.after(0, lambda: self.video_player.set_match_points([], [], None))
            return
        
        # Apply RANSAC filter
        sat_points_filtered, video_points_filtered, H = self.ransac_filter(
            sat_points_raw, video_points_raw, threshold=self.ransac_threshold
        )
        
        print(f"Frame {frame_num}: Raw={len(video_points_raw)} -> Filtered={len(video_points_filtered)}")
        
        if len(video_points_filtered) >= 4:
            self.root.after(0, lambda: self.video_player.set_match_points(
                video_points_filtered, sat_points_filtered, H
            ))
    
    def match_frame(self, frame):
        """
        Match a single video frame with satellite image.
        Returns: (video_points, satellite_points) - matched point coordinates
        """
        h, w = frame.shape[:2]
        
        # Resize for speed
        target = 480
        if max(h, w) > target:
            scale = target / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            small = cv2.resize(frame, (new_w, new_h))
            scale_x = w / new_w
            scale_y = h / new_h
        else:
            small = frame
            scale_x = 1
            scale_y = 1
        
        # Extract features from frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        tensor = transforms.ToTensor()(rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            frame_feats = self.extractor.extract(tensor)
        
        # Match with satellite using LightGlue
        with torch.no_grad():
            matches = self.matcher({"image0": self.satellite_feats, "image1": frame_feats})
        
        matches = rbd(matches)
        
        kpts_sat = self.satellite_feats["keypoints"][0].cpu().numpy()
        kpts_frame = frame_feats["keypoints"][0].cpu().numpy()
        matches_idx = matches["matches"].cpu().numpy()
        
        if len(matches_idx) < 4:
            return [], []
        
        # Get matched points
        sat_points = kpts_sat[matches_idx[:, 0]]
        frame_points = kpts_frame[matches_idx[:, 1]]
        
        # Scale back to original frame size
        frame_points_original = frame_points * [scale_x, scale_y]
        
        return frame_points_original, sat_points
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()