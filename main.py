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
        self.play_thread = None
        self.stop_thread = False
        self.current_photo = None
        
        #Данные
        self.match_points = []
        self.satellite_match_points = []
        self.homography_matrix = None
        
        self.update_display()
    
    #Загрузка видео (открытие чз OpenCv)
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
    
    #Отрисовка линий на комбинированном изображении с видео
    def draw_combined_with_lines(self, video_frame, sat_image, video_points, sat_points, H=None, max_lines=20):

        #Конверт спутника из PIL в numpy
        sat_np = np.array(sat_image.convert('RGB'))
        sat_np = cv2.cvtColor(sat_np, cv2.COLOR_RGB2BGR)
        
        #Ресайз спутник для метча с размерами видео
        sat_h, sat_w = sat_np.shape[:2]
        video_h, video_w = video_frame.shape[:2]
        scale = video_h / sat_h
        new_sat_w = int(sat_w * scale)
        sat_resized = cv2.resize(sat_np, (new_sat_w, video_h))
        
        #Создание комбо фото и видео
        combined = np.zeros((video_h, new_sat_w + video_w, 3), dtype=np.uint8)
        combined[:, :new_sat_w] = sat_resized
        combined[:, new_sat_w:] = video_frame
        
        #Скейлим ключевые у спутника
        if len(sat_points) > 0:
            sat_points_scaled = sat_points * [scale, scale]
        else:
            sat_points_scaled = []
        
        #Рисуем линии 
        num_lines = min(len(video_points), max_lines)

        for i in range(num_lines):

            if i < len(sat_points_scaled) and i < len(video_points):

                x1, y1 = sat_points_scaled[i]
                x2, y2 = video_points[i]
                
                cv2.circle(combined, (int(x1), int(y1)), 5, (0, 255, 0), -1)
                cv2.circle(combined, (int(x2 + new_sat_w), int(y2)), 5, (0, 0, 255), -1)
                cv2.line(combined, (int(x1), int(y1)), (int(x2 + new_sat_w), int(y2)), (255, 0, 0), 2)
        
        #H - гомографическая матрица
        if H is not None and len(sat_points_scaled) >= 4:
            cv2.putText(combined, "RANSAC: OK", (new_sat_w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return combined
    
    #Показываем кокнретный кадр на экране 
    def show_frame(self, frame_num):
  
        if self.cap is None:
            return
        
        #Читаем кадр из видео
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        
        if not ret:
            return
        
        #Если есть точки для отрисовки, вызываем рисову линий в комбо
        if len(self.match_points) >= 4 and len(self.satellite_match_points) >= 4 and self.parent_app.satellite_img is not None:
            combined = self.draw_combined_with_lines(
                frame, self.parent_app.satellite_img,
                self.match_points, self.satellite_match_points,
                self.homography_matrix, max_lines=20
            )
            display_frame = combined
        else:
            display_frame = frame
            for (x, y) in self.match_points:
                cv2.circle(display_frame, (int(x), int(y)), 4, (0, 255, 0), -1)
        
        # Конвертация для дисплея
        display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
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
    
    #Главный цикл воспроизведения
    def update_display(self):

        #Увеличиваем номер кадра если видео играет
        if self.is_playing and self.cap is not None:
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                self.current_frame = 0
            
            self.show_frame(self.current_frame)
            if self.total_frames > 0:
                self.seek_scale.set(int(self.current_frame / self.total_frames * 100))
            
            #Обрабатываем каждый 5 кадр (%5)
            if self.on_frame_callback and self.current_frame % 5 == 0:
                self.on_frame_callback(self.current_frame)
        
        self.parent.after(33, self.update_display)
    
    def set_match_points(self, video_points, satellite_points, H=None):

        self.match_points = video_points
        self.satellite_match_points = satellite_points
        self.homography_matrix = H

        if self.cap is not None:
            self.show_frame(self.current_frame)
    
    #Функции плеера
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
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BPLA - Satellite to Drone Matching")
        self.root.geometry("1200x750")
        
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        left_frame = tk.LabelFrame(self.root, text="Satellite Image", font=("Arial", 10, "bold"))
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        
        self.photo_canvas = tk.Canvas(left_frame, bg='gray', highlightthickness=0)
        self.photo_canvas.grid(row=0, column=0, sticky="nsew")
        
        right_frame = tk.LabelFrame(self.root, text="Drone Video", font=("Arial", 10, "bold"))
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        self.video_player = CustomVideoPlayer(right_frame, self, width=640, height=480, on_frame_callback=self.process_frame_background)
        
        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10)
        
        self.btn_load_photo = tk.Button(btn_frame, text="Load Satellite Photo", command=self.load_photo, width=18)
        self.btn_load_photo.pack(side=tk.LEFT, padx=5)
        
        self.btn_load_video = tk.Button(btn_frame, text="Load Video", command=self.load_video, width=12)
        self.btn_load_video.pack(side=tk.LEFT, padx=5)
        
        # RANSAC threshold слайдер
        tk.Label(btn_frame, text="RANSAC threshold:").pack(side=tk.LEFT, padx=(20,5))
        self.ransac_slider = tk.Scale(btn_frame, from_=1.0, to=10.0, orient=tk.HORIZONTAL,
                                       resolution=0.5, length=120, command=self.update_ransac_threshold)
        self.ransac_slider.set(3.0)
        self.ransac_slider.pack(side=tk.LEFT, padx=5)
        
        # Слайдер макс расстояния от центра до группы точек
        tk.Label(btn_frame, text="Max scatter (px):").pack(side=tk.LEFT, padx=(20,5))
        self.scatter_slider = tk.Scale(btn_frame, from_=50, to=300, orient=tk.HORIZONTAL,
                                        resolution=10, length=120, command=self.update_scatter_threshold)
        self.scatter_slider.set(150)
        self.scatter_slider.pack(side=tk.LEFT, padx=5)
        
        #Коэф уверенности
        tk.Label(btn_frame, text="Confidence:").pack(side=tk.LEFT, padx=(20,5))
        self.conf_slider = tk.Scale(btn_frame, from_=0.5, to=0.95, orient=tk.HORIZONTAL,
                                     resolution=0.05, length=120, command=self.update_conf_threshold)
        self.conf_slider.set(0.75)
        self.conf_slider.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(btn_frame, text="Ready", font=("Arial", 9))
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        #AI
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = None
        self.matcher = None
        self.satellite_feats = None
        self.satellite_img = None
        self.satellite_keypoints = None
        
        #Параметры по умолчанию
        self.ransac_threshold = 3.0
        self.max_scatter_distance = 150
        self.confidence_threshold = 0.75
        
        #Мэтч поток
        self.matching_thread = None
        self.last_frame_processed = -1
        
        self.load_ai()
    
    def update_ransac_threshold(self, value):
        self.ransac_threshold = float(value)
        self.status_label.config(text=f"RANSAC: {self.ransac_threshold}")
    
    def update_scatter_threshold(self, value):
        self.max_scatter_distance = float(value)
        self.status_label.config(text=f"Scatter: {self.max_scatter_distance}px")
    
    def update_conf_threshold(self, value):
        self.confidence_threshold = float(value)
        self.status_label.config(text=f"Confidence: {self.confidence_threshold}")
    
    #Загрузка иишек
    def load_ai(self):
        self.status_label.config(text="Loading AI...")
        self.root.update()
        
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        
        self.status_label.config(text=f"AI ready on {self.device}")
        print(f"AI loaded on {self.device}")
    
    #Загрузка фото
    def load_photo(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if not path:
            return
        
        self.satellite_img = Image.open(path)
        print(f"Loaded photo: {path}, size: {self.satellite_img.size}")
        
        self.status_label.config(text="Extracting satellite features...")
        self.root.update()
        
        #Извлечение ключей
        self.satellite_feats = self.extract_features(self.satellite_img)
        self.satellite_keypoints = self.extract_keypoints(self.satellite_img)
        
        self.display_photo_with_keypoints(self.satellite_keypoints)
        
        self.status_label.config(text=f"Satellite ready: {len(self.satellite_keypoints)} keypoints")
        print(f"Found {len(self.satellite_keypoints)} keypoints")
    
    #Ф-ция извлечения признаков сопоставления (ключи и дескрипторы)
    def extract_features(self, img):

        img_np = np.array(img.convert('RGB'))
        tensor = transforms.ToTensor()(img_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.extractor.extract(tensor)

        return feats
    
    #Ф-ция извлечения ключей
    def extract_keypoints(self, img):

        img_np = np.array(img.convert('RGB'))
        tensor = transforms.ToTensor()(img_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.extractor.extract(tensor)
        feats = rbd(feats)

        return feats['keypoints'].cpu().numpy()
    
    #Отображаем фото с добавленными поверх ключами (максимум поставили 100)
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
    
    #Загрузка видео
    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm")])
        if not path:
            return
        
        self.video_player.load(path)
        self.status_label.config(text="Video loaded. Click Play to start")
    
    #Фоновая обработка. Вызывается из плеера каждый 5 кадр
    def process_frame_background(self, frame_num):


        if self.satellite_feats is None:
            return
        
        if frame_num == self.last_frame_processed:
            return
        
        self.last_frame_processed = frame_num
        
        #Запускат в отдельном потоке match frame in bg
        if self.matching_thread is None or not self.matching_thread.is_alive():
            self.matching_thread = threading.Thread(
                target=self.match_frame_in_background,
                args=(frame_num,),
                daemon=True
            )
            self.matching_thread.start()
    
    #Ранзак фильтрация
    def ransac_filter(self, pts_sat, pts_frame, threshold=3.0, max_iter=2000):

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
    
    def check_points_clustering(self, points, max_distance=150):

        if len(points) < 3:
            return False
        
        # Calculate center of points
        center = np.mean(points, axis=0)
        
        # Calculate average distance from center
        distances = np.linalg.norm(points - center, axis=1)
        mean_distance = np.mean(distances)
        
        # Check if points are too scattered
        return mean_distance < max_distance
    
    def match_frame_in_background(self, frame_num):
        """Background matching with RANSAC and clustering check"""
        cap = cv2.VideoCapture(self.video_player.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return
        
        video_points_raw, sat_points_raw = self.match_frame(frame)
        
        if len(video_points_raw) < 8:
            self.root.after(0, lambda: self.video_player.set_match_points([], [], None))
            return
        
        # Apply RANSAC filter
        sat_points_filt, video_points_filt, H = self.ransac_filter(
            sat_points_raw, video_points_raw, threshold=self.ransac_threshold
        )
        
        # Additional check: points should be clustered
        if len(video_points_filt) >= 4:
            # Check scatter distance
            is_clustered = self.check_points_clustering(video_points_filt, self.max_scatter_distance)
            
            if not is_clustered:
                print(f"Frame {frame_num}: Points too scattered - rejecting")
                self.root.after(0, lambda: self.video_player.set_match_points([], [], None))
                return
            
            # Also check satellite points clustering
            is_sat_clustered = self.check_points_clustering(sat_points_filt, self.max_scatter_distance * 2)
            
            if not is_sat_clustered:
                print(f"Frame {frame_num}: Satellite points too scattered - rejecting")
                self.root.after(0, lambda: self.video_player.set_match_points([], [], None))
                return
            
            print(f"Frame {frame_num}: ACCEPTED {len(video_points_filt)} matches (scatter: clustered)")
            self.root.after(0, lambda: self.video_player.set_match_points(
                video_points_filt, sat_points_filt, H
            ))
        else:
            print(f"Frame {frame_num}: Not enough matches after RANSAC: {len(video_points_filt)}")
            self.root.after(0, lambda: self.video_player.set_match_points([], [], None))
    
    def match_frame(self, frame):
        """Match frame with satellite using SuperPoint and LightGlue"""
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
        
        # Match with satellite
        with torch.no_grad():
            matches = self.matcher({"image0": self.satellite_feats, "image1": frame_feats})
        
        matches = rbd(matches)
        
        kpts_sat = self.satellite_feats["keypoints"][0].cpu().numpy()
        kpts_frame = frame_feats["keypoints"][0].cpu().numpy()
        matches_idx = matches["matches"].cpu().numpy()
        scores = matches["scores"].cpu().numpy()
        
        if len(matches_idx) < 4:
            return [], []
        
        # Filter by confidence threshold
        conf_mask = scores > self.confidence_threshold
        matches_idx = matches_idx[conf_mask]
        
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