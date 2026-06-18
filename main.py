# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, Toplevel
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from lightglue import SuperPoint, LightGlue
from lightglue.utils import rbd
import threading
import time


class CustomVideoPlayer:
    """Кастомный видеоплеер с возможностью отображения точек поверх видео"""
    
    def __init__(self, parent, app, width=640, height=480, on_frame_callback=None):
        self.parent = parent
        self.parent_app = app
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
        
        # Состояние видео
        self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.fps = 30
        self.current_frame = 0
        self.is_playing = False
        self.current_photo = None
        
        # Данные для отрисовки совпадений
        self.match_points = []
        self.satellite_match_points = []
        self.homography_matrix = None
        
        # Флаг новых точек для отрисовки
        self.new_matches_available = False
        self.last_displayed_frame = -1
        
        self.update_display()
    
    def load(self, video_path):
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
    
    def show_frame(self, frame_num):
        if self.cap is None:
            return
        
        # Не перерисовываем тот же кадр повторно
        if frame_num == self.last_displayed_frame and not self.new_matches_available:
            return
        self.last_displayed_frame = frame_num
        self.new_matches_available = False
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        
        if not ret:
            return
        
        # Рисуем точки прямо на кадре
        if len(self.match_points) >= 4:
            for (x, y) in self.match_points[:30]:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Matches: {len(self.match_points)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Если есть совпадения и открыто окно сопоставления — обновляем его
        if len(self.match_points) >= 4 and self.parent_app.match_window is not None:
            self.parent_app.update_match_window(frame, self.match_points, self.satellite_match_points)
        
        display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
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
        
        img = Image.fromarray(display_rgb)
        self.current_photo = ImageTk.PhotoImage(img)
        
        self.canvas.delete("all")
        x = (canvas_w - new_w) // 2 if canvas_w > 1 else 0
        y = (canvas_h - new_h) // 2 if canvas_h > 1 else 0
        self.canvas.create_image(x, y, image=self.current_photo, anchor=tk.NW)
        
        current_time = frame_num / self.fps if self.fps > 0 else 0
        total_time = self.total_frames / self.fps if self.fps > 0 else 0
        self.time_label.config(text=f"{self._format_time(current_time)} / {self._format_time(total_time)}")
    
    def _format_time(self, seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"
    
    def update_display(self):
        if self.is_playing and self.cap is not None:
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                self.current_frame = 0
            
            self.show_frame(self.current_frame)
            if self.total_frames > 0:
                self.seek_scale.set(int(self.current_frame / self.total_frames * 100))
            
            # Обрабатываем каждый 15-й кадр
            if self.on_frame_callback and self.current_frame % 15 == 0:
                self.on_frame_callback(self.current_frame)
        
        self.parent.after(33, self.update_display)
    
    def set_match_points(self, video_points, satellite_points, H=None):
        self.match_points = video_points
        self.satellite_match_points = satellite_points
        self.homography_matrix = H
        self.new_matches_available = True
        if self.cap is not None:
            self.show_frame(self.current_frame)
    
    def play(self):
        if self.cap is not None and not self.is_playing:
            self.is_playing = True
    
    def pause(self):
        self.is_playing = False
    
    def stop(self):
        self.is_playing = False
        self.current_frame = 0
        self.show_frame(0)
        self.seek_scale.set(0)
    
    def seek(self, value):
        if self.cap is None or self.total_frames <= 0:
            return
        self.current_frame = int(int(value) / 100 * self.total_frames)
        self.show_frame(self.current_frame)
    
    def close(self):
        self.is_playing = False
        if self.cap:
            self.cap.release()


class App:
    """Главный класс приложения BPLA - сопоставление спутника и видео с дрона"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BPLA - Satellite to Drone Matching")
        self.root.geometry("1500x850")
        
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # ---- Левая панель: спутник ----
        left_frame = tk.LabelFrame(self.root, text="Satellite Image", font=("Arial", 10, "bold"))
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        
        self.photo_canvas = tk.Canvas(left_frame, bg='gray', highlightthickness=0)
        self.photo_canvas.grid(row=0, column=0, sticky="nsew")
        
        # ---- Правая панель: видео ----
        right_frame = tk.LabelFrame(self.root, text="Drone Video", font=("Arial", 10, "bold"))
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        self.video_player = CustomVideoPlayer(right_frame, self, width=640, height=480, on_frame_callback=self.process_frame_background)
        
        # ---- Нижняя панель: управление ----
        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10)
        
        self.btn_load_photo = tk.Button(btn_frame, text="Load Satellite Photo", command=self.load_photo, width=18)
        self.btn_load_photo.pack(side=tk.LEFT, padx=5)
        
        self.btn_load_video = tk.Button(btn_frame, text="Load Video", command=self.load_video, width=12)
        self.btn_load_video.pack(side=tk.LEFT, padx=5)
        
        self.btn_show_matches = tk.Button(btn_frame, text="Show Matches Window", command=self.open_match_window, width=18, bg='lightblue')
        self.btn_show_matches.pack(side=tk.LEFT, padx=5)
        
        # ---- НОВЫЕ НАСТРОЙКИ ИЗ КИТАЙСКОГО РЕПОЗИТОРИЯ ----
        # 1. Rotation search (поиск лучшего угла)
        self.rotation_var = tk.IntVar(value=0)
        self.rotation_check = tk.Checkbutton(
            btn_frame, 
            text="Rotation search",
            variable=self.rotation_var,
            command=self.toggle_rotation
        )
        self.rotation_check.pack(side=tk.LEFT, padx=(20,5))
        
        # 2. Количество углов для поиска
        tk.Label(btn_frame, text="Angles:").pack(side=tk.LEFT, padx=(10,2))
        self.angles_slider = tk.Scale(
            btn_frame, from_=3, to=9, orient=tk.HORIZONTAL,
            resolution=2, length=80, command=self.update_angles
        )
        self.angles_slider.set(5)
        self.angles_slider.pack(side=tk.LEFT, padx=2)

        tk.Label(btn_frame, text="RANSAC:").pack(side=tk.LEFT, padx=(10,2))
        self.ransac_slider = tk.Scale(btn_frame, from_=1.0, to=10.0, orient=tk.HORIZONTAL,
                                       resolution=0.5, length=100, command=self.update_ransac_threshold)
        self.ransac_slider.set(4.0)
        self.ransac_slider.pack(side=tk.LEFT, padx=2)
        
        tk.Label(btn_frame, text="Scatter:").pack(side=tk.LEFT, padx=(10,2))
        self.scatter_slider = tk.Scale(btn_frame, from_=50, to=600, orient=tk.HORIZONTAL,
                                        resolution=10, length=100, command=self.update_scatter_threshold)
        self.scatter_slider.set(300)
        self.scatter_slider.pack(side=tk.LEFT, padx=2)
        
        tk.Label(btn_frame, text="Confidence:").pack(side=tk.LEFT, padx=(10,2))
        self.conf_slider = tk.Scale(btn_frame, from_=0.3, to=0.95, orient=tk.HORIZONTAL,
                                     resolution=0.05, length=100, command=self.update_conf_threshold)
        self.conf_slider.set(0.65)
        self.conf_slider.pack(side=tk.LEFT, padx=2)
        
        tk.Label(btn_frame, text="Skip:").pack(side=tk.LEFT, padx=(10,2))
        self.skip_slider = tk.Scale(btn_frame, from_=2, to=30, orient=tk.HORIZONTAL,
                                     resolution=1, length=80, command=self.update_skip)
        self.skip_slider.set(15)
        self.skip_slider.pack(side=tk.LEFT, padx=2)
        
        tk.Label(btn_frame, text="Resize:").pack(side=tk.LEFT, padx=(10,2))
        self.resize_slider = tk.Scale(btn_frame, from_=160, to=640, orient=tk.HORIZONTAL,
                                       resolution=32, length=80, command=self.update_resize)
        self.resize_slider.set(256)
        self.resize_slider.pack(side=tk.LEFT, padx=2)
        
        self.status_label = tk.Label(btn_frame, text="Ready", font=("Arial", 9))
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # AI модели
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = None
        self.matcher = None
        self.satellite_feats = None
        self.satellite_img = None
        self.satellite_keypoints = None
        
        # Параметры по умолчанию
        self.ransac_threshold = 4.0
        self.max_scatter_distance = 300
        self.confidence_threshold = 0.65
        self.frame_skip = 15
        self.target_size = 256
        self.num_angles = 5
        self.rotation_search_enabled = False
        
        # Окно сопоставления
        self.match_window = None
        self.match_canvas = None
        self.match_photo = None
        
        # Поток для матчинга
        self.matching_thread = None
        self.last_frame_processed = -1
        
        self.load_ai()
    
    # ----------ИЗ КИТАЙСКОГО РЕПОЗИТОРИЯ ----------
    
    def toggle_rotation(self):
        """Включает/выключает поиск углов"""
        self.rotation_search_enabled = self.rotation_var.get() == 1
        state = "ON" if self.rotation_search_enabled else "OFF"
        self.status_label.config(text=f"Rotation: {state}")
        print(f"Rotation search: {state}")
    
    def update_angles(self, value):
        """Обновляет количество углов для поиска"""
        self.num_angles = int(value)
        self.status_label.config(text=f"Angles: {self.num_angles}")
    
    def rotate_image(self, image, angle):
        """Поворачивает изображение на заданный угол (в градусах)"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def find_best_rotation(self, frame):
        """Пробует несколько углов и возвращает лучший"""
        # Генерируем углы в зависимости от количества
        if self.num_angles == 3:
            angles = [-20, 0, 20]
        elif self.num_angles == 5:
            angles = [-25, -12, 0, 12, 25]
        elif self.num_angles == 7:
            angles = [-30, -20, -10, 0, 10, 20, 30]
        else:  # 9 углов
            angles = [-30, -22, -15, -7, 0, 7, 15, 22, 30]
        
        best_video_points = []
        best_sat_points = []
        best_count = 0
        best_angle = 0
        
        for angle in angles:
            rotated_frame = self.rotate_image(frame, angle)
            video_points, sat_points = self.match_frame(rotated_frame)
            
            if len(video_points) > best_count:
                best_count = len(video_points)
                best_video_points = video_points
                best_sat_points = sat_points
                best_angle = angle
        
        if best_count >= 4:
            print(f"Best angle: {best_angle}°, matches: {best_count}")
        
        return best_video_points, best_sat_points, best_angle
    
    def update_ransac_threshold(self, value):
        self.ransac_threshold = float(value)
        self.status_label.config(text=f"RANSAC: {self.ransac_threshold}")
    
    def update_scatter_threshold(self, value):
        self.max_scatter_distance = float(value)
        self.status_label.config(text=f"Scatter: {self.max_scatter_distance}px")
    
    def update_conf_threshold(self, value):
        self.confidence_threshold = float(value)
        self.status_label.config(text=f"Confidence: {self.confidence_threshold}")
    
    def update_skip(self, value):
        self.frame_skip = int(value)
        self.status_label.config(text=f"Skip: {self.frame_skip}")
    
    def update_resize(self, value):
        self.target_size = int(value)
        self.status_label.config(text=f"Resize: {self.target_size}")
    
    # --------------------- ОКНО СОПОСТАВЛЕНИЯ ---------------------
    
    def open_match_window(self):
        if self.match_window is not None and self.match_window.winfo_exists():
            self.match_window.lift()
            return
        
        self.match_window = Toplevel(self.root)
        self.match_window.title("Matches: Satellite ↔ Drone")
        self.match_window.geometry("1200x700")
        
        self.match_canvas = tk.Canvas(self.match_window, bg='black')
        self.match_canvas.pack(fill=tk.BOTH, expand=True)
        
        info_label = tk.Label(self.match_window, text="Matches will appear here when video is playing", font=("Arial", 10))
        info_label.pack(side=tk.BOTTOM, pady=5)
        
        self.match_window.protocol("WM_DELETE_WINDOW", self.close_match_window)
    
    def close_match_window(self):
        if self.match_window is not None:
            self.match_window.destroy()
            self.match_window = None
            self.match_canvas = None
    
    def update_match_window(self, video_frame, video_points, sat_points):
        if self.match_canvas is None or self.satellite_img is None:
            return
        
        sat_np = np.array(self.satellite_img.convert('RGB'))
        sat_np = cv2.cvtColor(sat_np, cv2.COLOR_RGB2BGR)
        
        sat_h, sat_w = sat_np.shape[:2]
        video_h, video_w = video_frame.shape[:2]
        scale = video_h / sat_h
        new_sat_w = int(sat_w * scale)
        sat_resized = cv2.resize(sat_np, (new_sat_w, video_h))
        
        combined = np.zeros((video_h, new_sat_w + video_w, 3), dtype=np.uint8)
        combined[:, :new_sat_w] = sat_resized
        combined[:, new_sat_w:] = video_frame
        
        sat_points_scaled = sat_points * [scale, scale]
        
        num_lines = min(len(video_points), 30)
        for i in range(num_lines):
            if i < len(sat_points_scaled):
                x1, y1 = sat_points_scaled[i]
                x2, y2 = video_points[i]
                
                cv2.circle(combined, (int(x1), int(y1)), 6, (0, 255, 0), -1)
                cv2.circle(combined, (int(x2 + new_sat_w), int(y2)), 6, (0, 0, 255), -1)
                cv2.line(combined, (int(x1), int(y1)), (int(x2 + new_sat_w), int(y2)), (255, 0, 0), 3)
        
        cv2.putText(combined, "SATELLITE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(combined, "DRONE", (new_sat_w + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(combined, f"Matches: {len(video_points)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        
        canvas_w = self.match_canvas.winfo_width()
        canvas_h = self.match_canvas.winfo_height()
        
        if canvas_w <= 1:
            canvas_w = 1200
            canvas_h = 650
        
        h, w = combined_rgb.shape[:2]
        scale_disp = min(canvas_w / w, canvas_h / h)
        new_w = int(w * scale_disp)
        new_h = int(h * scale_disp)
        combined_rgb = cv2.resize(combined_rgb, (new_w, new_h))
        
        img = Image.fromarray(combined_rgb)
        self.match_photo = ImageTk.PhotoImage(img)
        
        self.match_canvas.delete("all")
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        self.match_canvas.create_image(x, y, image=self.match_photo, anchor=tk.NW)
    
    # --------------------- ОСНОВНЫЕ ФУНКЦИИ ---------------------
    
    def load_ai(self):
        self.status_label.config(text="Loading AI...")
        self.root.update()
        
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        
        self.status_label.config(text=f"AI ready on {self.device}")
        print(f"AI loaded on {self.device}")
    
    def load_photo(self):
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
        img_np = np.array(img.convert('RGB'))
        tensor = transforms.ToTensor()(img_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.extractor.extract(tensor)
        return feats
    
    def extract_keypoints(self, img):
        img_np = np.array(img.convert('RGB'))
        tensor = transforms.ToTensor()(img_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.extractor.extract(tensor)
        feats = rbd(feats)
        return feats['keypoints'].cpu().numpy()
    
    def display_photo_with_keypoints(self, keypoints, max_points=100):
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
        path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm")])
        if not path:
            return
        
        self.video_player.load(path)
        self.status_label.config(text="Video loaded. Click Play to start")
    
    def process_frame_background(self, frame_num):
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
    
    def ransac_filter(self, pts_sat, pts_frame, threshold=4.0, max_iter=2000):
        if len(pts_sat) < 4:
            return pts_sat, pts_frame, None
        
        src_pts = pts_sat.reshape(-1, 1, 2).astype(np.float32)
        dst_pts = pts_frame.reshape(-1, 1, 2).astype(np.float32)
        
        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=threshold,
            maxIters=max_iter,
            confidence=0.995
        )
        
        if H is not None and mask is not None:
            mask = mask.ravel().astype(bool)
            filtered_sat = pts_sat[mask]
            filtered_frame = pts_frame[mask]
            return filtered_sat, filtered_frame, H
        else:
            return pts_sat, pts_frame, None
    
    def check_points_clustering(self, points, max_distance=300):
        if len(points) < 3:
            return False
        
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        mean_distance = np.mean(distances)
        
        return mean_distance < max_distance
    
    def match_frame_in_background(self, frame_num):
        cap = cv2.VideoCapture(self.video_player.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return
        
        # === ПОИСК ЛУЧШЕГО УГЛА (НОВЫЙ КОД) ===
        if self.rotation_search_enabled:
            video_points_raw, sat_points_raw, best_angle = self.find_best_rotation(frame)
            if len(video_points_raw) < 4:
                video_points_raw, sat_points_raw = self.match_frame(frame)
        else:
            video_points_raw, sat_points_raw = self.match_frame(frame)
        
        if len(video_points_raw) < 8:
            self.root.after(0, lambda: self.video_player.set_match_points([], [], None))
            return
        
        # RANSAC фильтрация
        sat_points_filt, video_points_filt, H = self.ransac_filter(
            sat_points_raw, video_points_raw, threshold=self.ransac_threshold
        )
        
        if len(video_points_filt) >= 4:
            is_clustered = self.check_points_clustering(video_points_filt, self.max_scatter_distance)
            
            if not is_clustered:
                print(f"Frame {frame_num}: Points too scattered - rejecting")
                self.root.after(0, lambda: self.video_player.set_match_points([], [], None))
                return
            
            is_sat_clustered = self.check_points_clustering(sat_points_filt, self.max_scatter_distance * 2)
            
            if not is_sat_clustered:
                print(f"Frame {frame_num}: Satellite points too scattered - rejecting")
                self.root.after(0, lambda: self.video_player.set_match_points([], [], None))
                return
            
            print(f"Frame {frame_num}: ACCEPTED {len(video_points_filt)} matches")
            self.root.after(0, lambda: self.video_player.set_match_points(
                video_points_filt, sat_points_filt, H
            ))
        else:
            print(f"Frame {frame_num}: Not enough matches after RANSAC: {len(video_points_filt)}")
            self.root.after(0, lambda: self.video_player.set_match_points([], [], None))
    
    def match_frame(self, frame):
        h, w = frame.shape[:2]
        
        target = self.target_size
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
        
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        tensor = transforms.ToTensor()(rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            frame_feats = self.extractor.extract(tensor)
        
        with torch.no_grad():
            matches = self.matcher({"image0": self.satellite_feats, "image1": frame_feats})
        
        matches = rbd(matches)
        
        kpts_sat = self.satellite_feats["keypoints"][0].cpu().numpy()
        kpts_frame = frame_feats["keypoints"][0].cpu().numpy()
        matches_idx = matches["matches"].cpu().numpy()
        scores = matches["scores"].cpu().numpy()
        
        if len(matches_idx) < 4:
            return [], []
        
        conf_mask = scores > self.confidence_threshold
        matches_idx = matches_idx[conf_mask]
        
        if len(matches_idx) < 4:
            return [], []
        
        sat_points = kpts_sat[matches_idx[:, 0]]
        frame_points = kpts_frame[matches_idx[:, 1]]
        
        frame_points_original = frame_points * [scale_x, scale_y]
        
        return frame_points_original, sat_points
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()