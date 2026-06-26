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
from collections import deque
import av


class CustomVideoPlayer:
    
    def __init__(self, parent, app, width=640, height=480, on_frame_callback=None):

        self.parent = parent
        self.parent_app = app
        self.width = width
        self.height = height
        self.on_frame_callback = on_frame_callback
        
        # Основной фрейм для видеоплеера
        self.frame = tk.Frame(parent, bg='black')
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas для отображения видео
        self.canvas = tk.Canvas(self.frame, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Фрейм с кнопками управления
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
        
        #ДВА ИСТОЧНИКА ВИДЕО
        self.cap_opencv = None          # Для перемотки и информации (OpenCV)
        self.container_av = None        # Для GPU-декодирования (PyAV)
        self.video_stream = None        # Видеопоток из PyAV
        self.video_path = None          # Путь к видеофайлу
        self.total_frames = 0           # Общее количество кадров
        self.fps = 30                   # Кадров в секунду
        self.current_frame = 0          # Текущий кадр
        self.is_playing = False         # Флаг воспроизведения
        self.current_photo = None       # Текущее отображаемое изображение
        self.use_gpu = False            # Флаг использования GPU
        
        # Данные для отрисовки совпадений
        self.match_points = []              # Точки совпадений на видео
        self.satellite_match_points = []    # Соответствующие точки на спутнике
        self.homography_matrix = None       # Матрица гомографии
        self.last_match_frame = 0           # Номер кадра с последними совпадениями
        
        # Флаг новых точек для отрисовки
        self.new_matches_available = False
        self.last_displayed_frame = -1
        
        #КЭШ КАДРОВ (оригинальный размер)
        self.frame_cache = {}               # Словарь для кэширования кадров
        self.cache_size = 200               # Максимальный размер кэша
        self.cache_lock = threading.Lock()  # Блокировка для безопасного доступа к кэшу
        
        #БУФЕР ПРЕДЗАГРУЗКИ (оригинальный размер)
        self.buffer_thread = None           # Поток для предзагрузки кадров
        self.buffer_running = False         # Флаг работы буфера
        self.buffer_queue = deque(maxlen=60) # Очередь буфера
        self.buffer_lock = threading.Lock() # Блокировка для буфера
        
        self.update_display()
    
    def load(self, video_path):
        """
        Загружает видеофайл.
        """
        self.video_path = video_path
        
        # 1. ОТКРЫВАЕМ ЧЕРЕЗ OpenCV
        self.cap_opencv = cv2.VideoCapture(video_path)
        if self.cap_opencv.isOpened():
            self.total_frames = int(self.cap_opencv.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap_opencv.get(cv2.CAP_PROP_FPS)
        else:
            self.total_frames = 0
            self.fps = 30
        
        if self.fps <= 0:
            self.fps = 30
        
        print(f"Информация о видео из OpenCV: {self.total_frames} кадров, {self.fps:.1f} FPS")
        
        #2. ПЫТАЕМСЯ ОТКРЫТЬ ЧЕРЕЗ PyAV С GPU
        try:
            self.container_av = av.open(video_path)
            self.video_stream = self.container_av.streams.video[0]
            
            # Включаем аппаратное ускорение через CUDA
            try:
                codec = self.video_stream.codec_context
                codec.options = {
                    'hwaccel': 'cuda',           # Используем CUDA
                    'hwaccel_device': '0',        # Первая видеокарта
                    'hwaccel_output_format': 'cuda'  # Оставляем на GPU
                }
                self.use_gpu = True
                print("GPU ускорение ВКЛЮЧЕНО (CUDA/NVDEC)")

            except Exception as e:
                print(f"GPU ускорение не доступно: {e}")
                self.use_gpu = False
            
            print(f"Видео загружено: {self.total_frames} кадров, {self.fps:.1f} FPS, GPU: {self.use_gpu}")
            
            # Очищаем кэш и буфер
            with self.cache_lock:
                self.frame_cache.clear()

            with self.buffer_lock:
                self.buffer_queue.clear()
            
            # Запускаем поток предзагрузки
            self._start_buffer_thread()
            
            self.current_frame = 0
            self.show_frame(0)
            return True
            
        except Exception as e:

            print(f"Ошибка PyAV: {e}")
            return self._fallback_to_cpu(video_path)
    
    def _fallback_to_cpu(self, video_path):
        """
        Запасной вариант через OpenCV (CPU) если PyAV не работает.
        """

        print("Используем запасной вариант на CPU (OpenCV)")
        self.cap_opencv = cv2.VideoCapture(video_path)

        if not self.cap_opencv.isOpened():
            return False

        self.total_frames = int(self.cap_opencv.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap_opencv.get(cv2.CAP_PROP_FPS)

        if self.fps <= 0:
            self.fps = 30

        self.current_frame = 0
        self.show_frame(0)
        self.use_gpu = False

        return True
    
    def _start_buffer_thread(self):
        """Запускает поток для предзагрузки кадров в буфер."""

        if self.buffer_thread is not None and self.buffer_thread.is_alive():
            return
        
        self.buffer_running = True
        self.buffer_thread = threading.Thread(target=self._buffer_loop, daemon=True)
        self.buffer_thread.start()
    
    def _buffer_loop(self):
        """
        Заполняет буфер кадрами в фоновом потоке
        """
        try:
            container = av.open(self.video_path)
            stream = container.streams.video[0]
            
            # Включаем GPU для буфера
            try:
                codec = stream.codec_context
                codec.options = {
                    'hwaccel': 'cuda',
                    'hwaccel_device': '0'
                }
            except:
                pass
            
            frame_num = 0
            
            for packet in container.demux(stream):
                if not self.buffer_running:
                    break
                
                for frame in packet.decode():
                    if not self.buffer_running:
                        break
                    
                    # Конвертируем кадр в numpy (BGR для OpenCV)
                    try:
                        img = frame.to_ndarray(format='bgr24')

                    except:
                        img = frame.to_ndarray(format='bgr24')
                    
                    # Сохраняем в буфер в оригинальном размере
                    with self.buffer_lock:

                        if len(self.buffer_queue) < self.buffer_queue.maxlen:
                            self.buffer_queue.append((frame_num, img))

                        else:
                            time.sleep(0.01)
                            continue
                    
                    frame_num += 1
                    time.sleep(0.005)
            
            container.close()

        except Exception as e:
            print(f"Ошибка буфера: {e}")
    
    def _get_frame_from_opencv(self, frame_num):
        """
        Получает кадр через OpenCV
        """
        if self.cap_opencv is None:
            return None
        
        self.cap_opencv.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap_opencv.read()

        if ret:
            return frame

        return None
    
    def _get_cached_frame(self, frame_num):
        """
        Получает кадр из кэша, буфера или напрямую через OpenCV.

        """
        # 1. Проверяем кэш
        with self.cache_lock:
            if frame_num in self.frame_cache:
                return self.frame_cache[frame_num].copy()
        
        # 2. Проверяем буфер (PyAV GPU)
        with self.buffer_lock:

            for buffered_num, buffered_frame in self.buffer_queue:

                if buffered_num == frame_num:

                    with self.cache_lock:

                        if len(self.frame_cache) < self.cache_size:
                            self.frame_cache[frame_num] = buffered_frame.copy()

                    return buffered_frame.copy()
        
        # 3. Читаем через OpenCV (для перемотки)
        frame = self._get_frame_from_opencv(frame_num)
        if frame is not None:

            with self.cache_lock:
                if len(self.frame_cache) < self.cache_size:
                    self.frame_cache[frame_num] = frame.copy()

            return frame
        
        return None
    
    def show_frame(self, frame_num):
        """
        Отображает конкретный кадр с наложенными точками совпадений
        """
        if self.cap_opencv is None and self.container_av is None:
            return
        
        # Не перерисовываем тот же кадр без новых данных
        if frame_num == self.last_displayed_frame and not self.new_matches_available:
            return

        self.last_displayed_frame = frame_num
        self.new_matches_available = False
        
        # Получаем кадр
        frame = self._get_cached_frame(frame_num)
        if frame is None:
            return
        
        # Рисуем точки совпадений прямо на кадре
        display_frame = frame.copy()

        if len(self.match_points) >= 4:

            for (x, y) in self.match_points[:30]:
                cv2.circle(display_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(display_frame, f"Совпадений: {len(self.match_points)} (кадр {self.last_match_frame})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Если есть совпадения и открыто окно сопоставления — обновляем его
        if len(self.match_points) >= 4 and self.parent_app.match_window is not None:
            self.parent_app.update_match_window(
                display_frame, 
                self.match_points, 
                self.satellite_match_points,
                frame_num
            )
        
        # Конвертируем для отображения
        display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Получаем размеры Canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        # Масштабируем только для отображения
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
        
        # Отображаем на Canvas
        self.canvas.delete("all")

        x = (canvas_w - new_w) // 2 if canvas_w > 1 else 0
        y = (canvas_h - new_h) // 2 if canvas_h > 1 else 0
        self.canvas.create_image(x, y, image=self.current_photo, anchor=tk.NW)
        
        # Обновляем метку времени
        current_time = frame_num / self.fps if self.fps > 0 else 0
        total_time = self.total_frames / self.fps if self.fps > 0 else 0
        self.time_label.config(text=f"{self._format_time(current_time)} / {self._format_time(total_time)}")
    
    def _format_time(self, seconds):

        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"
    
    def update_display(self):
        """
        Главный цикл воспроизведения. Вызывается каждые 33 мс.
        Обновляет отображение и вызывает обработку кадров.
        """
        if self.is_playing:
            self.current_frame += 1
            if self.current_frame >= self.total_frames and self.total_frames > 0:
                self.current_frame = 0
            
            self.show_frame(self.current_frame)
            if self.total_frames > 0:
                self.seek_scale.set(int(self.current_frame / self.total_frames * 100))
            
            # Обрабатываем каждый 15-й кадр для поиска совпадений
            if self.on_frame_callback and self.current_frame % 15 == 0:
                self.on_frame_callback(self.current_frame)
        
        self.parent.after(33, self.update_display)
    
    def set_match_points(self, video_points, satellite_points, H=None):
        """
        Устанавливает точки совпадений для отрисовки.
        Вызывается из фонового потока обработки.
        """

        self.match_points = video_points
        self.satellite_match_points = satellite_points
        self.homography_matrix = H
        self.last_match_frame = self.current_frame
        self.new_matches_available = True
        self.show_frame(self.current_frame)
    
    def play(self):
        if not self.is_playing:
            self.is_playing = True
    
    def pause(self):
        self.is_playing = False
    
    def stop(self):
        self.is_playing = False
        self.current_frame = 0
        self.show_frame(0)
        self.seek_scale.set(0)
    
    def seek(self, value):
        """
        Перематывает видео на указанную позицию.
        """
        if self.total_frames <= 0:
            return
        self.current_frame = int(int(value) / 100 * self.total_frames)
        self.show_frame(self.current_frame)
    
    def close(self):

        self.is_playing = False
        self.buffer_running = False

        if self.container_av:
            self.container_av.close()

        if self.cap_opencv:
            self.cap_opencv.release()


class App:
    
    def __init__(self):

        self.root = tk.Tk()
        self.root.title("BPLA - Satellite to Drone Matching")
        self.root.geometry("1500x850")
        
        # Настройка сетки для растягивания
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
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
        
        self.btn_show_matches = tk.Button(btn_frame, text="Show Matches Window", command=self.open_match_window, width=18, bg='lightblue')
        self.btn_show_matches.pack(side=tk.LEFT, padx=5)
        
        # Поиск углов (Rotation search)
        self.rotation_var = tk.IntVar(value=0)
        self.rotation_check = tk.Checkbutton(
            btn_frame, 
            text="Rotation search",
            variable=self.rotation_var,
            command=self.toggle_rotation
        )
        self.rotation_check.pack(side=tk.LEFT, padx=(20,5))
        
        # Количество углов для поиска
        tk.Label(btn_frame, text="Angles:").pack(side=tk.LEFT, padx=(10,2))
        self.angles_slider = tk.Scale(
            btn_frame, from_=3, to=9, orient=tk.HORIZONTAL,
            resolution=2, length=80, command=self.update_angles
        )
        self.angles_slider.set(5)
        self.angles_slider.pack(side=tk.LEFT, padx=2)

        # RANSAC порог
        tk.Label(btn_frame, text="RANSAC:").pack(side=tk.LEFT, padx=(10,2))
        self.ransac_slider = tk.Scale(btn_frame, from_=1.0, to=10.0, orient=tk.HORIZONTAL,
                                       resolution=0.5, length=100, command=self.update_ransac_threshold)
        self.ransac_slider.set(4.0)
        self.ransac_slider.pack(side=tk.LEFT, padx=2)
        
        # Scatter (разбросанность точек)
        tk.Label(btn_frame, text="Scatter:").pack(side=tk.LEFT, padx=(10,2))
        self.scatter_slider = tk.Scale(btn_frame, from_=50, to=600, orient=tk.HORIZONTAL,
                                        resolution=10, length=100, command=self.update_scatter_threshold)
        self.scatter_slider.set(300)
        self.scatter_slider.pack(side=tk.LEFT, padx=2)
        
        # Confidence (уверенность LightGlue)
        tk.Label(btn_frame, text="Confidence:").pack(side=tk.LEFT, padx=(10,2))
        self.conf_slider = tk.Scale(btn_frame, from_=0.3, to=0.95, orient=tk.HORIZONTAL,
                                     resolution=0.05, length=100, command=self.update_conf_threshold)
        self.conf_slider.set(0.65)
        self.conf_slider.pack(side=tk.LEFT, padx=2)
        
        # Frame Skip (пропуск кадров)
        tk.Label(btn_frame, text="Skip:").pack(side=tk.LEFT, padx=(10,2))
        self.skip_slider = tk.Scale(btn_frame, from_=2, to=30, orient=tk.HORIZONTAL,
                                     resolution=1, length=80, command=self.update_skip)
        self.skip_slider.set(15)
        self.skip_slider.pack(side=tk.LEFT, padx=2)
        
        # Resize (размер кадра для обработки)
        tk.Label(btn_frame, text="Resize:").pack(side=tk.LEFT, padx=(10,2))
        self.resize_slider = tk.Scale(btn_frame, from_=160, to=640, orient=tk.HORIZONTAL,
                                       resolution=32, length=80, command=self.update_resize)
        self.resize_slider.set(640)  
        self.resize_slider.pack(side=tk.LEFT, padx=2)
        
        self.status_label = tk.Label(btn_frame, text="Ready", font=("Arial", 9))
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # ---- AI модели ----
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
        self.target_size = 640
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
    
    def toggle_rotation(self):
        """Включает/выключает поиск оптимального угла поворота."""

        self.rotation_search_enabled = self.rotation_var.get() == 1
        state = "ON" if self.rotation_search_enabled else "OFF"
        self.status_label.config(text=f"Rotation: {state}")

        print(f"Rotation search: {state}")
    
    def update_angles(self, value):
        """Обновляет количество углов для поиска."""

        self.num_angles = int(value)
        self.status_label.config(text=f"Angles: {self.num_angles}")
    
    def rotate_image(self, image, angle):
        """
        Поворачивает изображение на заданный угол.
        """

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return rotated
    
    def find_best_rotation(self, frame):
        """
        Пробует несколько углов поворота и возвращает лучший
        """

        # Генерируем углы в зависимости от количества
        if self.num_angles == 3:
            angles = [-20, 0, 20]

        elif self.num_angles == 5:
            angles = [-25, -12, 0, 12, 25]

        elif self.num_angles == 7:
            angles = [-30, -20, -10, 0, 10, 20, 30]

        else:
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
            print(f"Лучший угол: {best_angle}°, совпадений: {best_count}")
        
        return best_video_points, best_sat_points, best_angle
    
    def update_ransac_threshold(self, value):
        """Обновляет порог RANSAC."""

        self.ransac_threshold = float(value)
        self.status_label.config(text=f"RANSAC: {self.ransac_threshold}")
    
    def update_scatter_threshold(self, value):
        """Обновляет порог разбросанности точек."""

        self.max_scatter_distance = float(value)
        self.status_label.config(text=f"Scatter: {self.max_scatter_distance}px")
    
    def update_conf_threshold(self, value):
        """Обновляет порог уверенности LightGlue."""

        self.confidence_threshold = float(value)
        self.status_label.config(text=f"Confidence: {self.confidence_threshold}")
    
    def update_skip(self, value):
        """Обновляет частоту обработки кадров."""

        self.frame_skip = int(value)
        self.status_label.config(text=f"Skip: {self.frame_skip}")
    
    def update_resize(self, value):
        """Обновляет размер кадра для обработки."""

        self.target_size = int(value)
        self.status_label.config(text=f"Resize: {self.target_size}")
    
    def open_match_window(self):
        """
        Открывает отдельное окно для отображения сопоставлений.
        Показывает спутник и видео рядом с линиями между точками.
        """
        if self.match_window is not None and self.match_window.winfo_exists():
            self.match_window.lift()
            return
        
        self.match_window = Toplevel(self.root)
        self.match_window.title("Matches: Satellite ↔ Drone")
        self.match_window.geometry("1200x700")
        
        self.match_canvas = tk.Canvas(self.match_window, bg='black')
        self.match_canvas.pack(fill=tk.BOTH, expand=True)
        
        info_label = tk.Label(self.match_window, text="Сопоставления появятся здесь во время воспроизведения видео", font=("Arial", 10))
        info_label.pack(side=tk.BOTTOM, pady=5)
        
        self.match_window.protocol("WM_DELETE_WINDOW", self.close_match_window)
    
    def close_match_window(self):
        """Закрывает окно сопоставления."""

        if self.match_window is not None:
            self.match_window.destroy()
            self.match_window = None
            self.match_canvas = None
    
    def update_match_window(self, video_frame, video_points, sat_points, frame_num):
        """
        Обновляет окно сопоставления: склеивает спутник и видео с линиями между точками.
        """

        if self.match_canvas is None or self.satellite_img is None:
            return
        
        # Конвертируем спутник в BGR
        sat_np = np.array(self.satellite_img.convert('RGB'))
        sat_np = cv2.cvtColor(sat_np, cv2.COLOR_RGB2BGR)
        
        sat_h, sat_w = sat_np.shape[:2]
        video_h, video_w = video_frame.shape[:2]
        
        # Масштабируем спутник под высоту видео
        scale = video_h / sat_h
        new_sat_w = int(sat_w * scale)
        sat_resized = cv2.resize(sat_np, (new_sat_w, video_h))
        
        # Склеиваем изображения
        combined = np.zeros((video_h, new_sat_w + video_w, 3), dtype=np.uint8)
        combined[:, :new_sat_w] = sat_resized
        combined[:, new_sat_w:] = video_frame
        
        # Масштабируем точки спутника
        sat_points_scaled = sat_points * [scale, scale]
        
        # Рисуем линии между точками
        num_lines = min(len(video_points), 30)

        for i in range(num_lines):
            if i < len(sat_points_scaled):
                x1, y1 = sat_points_scaled[i]
                x2, y2 = video_points[i]
                
                cv2.circle(combined, (int(x1), int(y1)), 6, (0, 255, 0), -1)
                cv2.circle(combined, (int(x2 + new_sat_w), int(y2)), 6, (0, 0, 255), -1)
                cv2.line(combined, (int(x1), int(y1)), (int(x2 + new_sat_w), int(y2)), (255, 0, 0), 3)
        
        # Добавляем подписи
        cv2.putText(combined, "SATELLITE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(combined, f"DRONE (кадр {frame_num})", (new_sat_w + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(combined, f"Совпадений: {len(video_points)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Отображаем в окне
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
    
    def load_ai(self):
        """Загружает нейросети SuperPoint и LightGlue."""

        self.status_label.config(text="Загрузка AI...")
        self.root.update()
        
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        
        self.status_label.config(text=f"AI готов на {self.device}")
        print(f"AI загружен на {self.device}")
    
    def load_photo(self):
        """Загружает спутниковое фото и извлекает ключевые точки."""

        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if not path:
            return
        
        self.satellite_img = Image.open(path)
        print(f"Загружено фото: {path}, размер: {self.satellite_img.size}")
        
        self.status_label.config(text="Извлечение признаков спутника...")
        self.root.update()
        
        # Извлекаем признаки для сопоставления
        self.satellite_feats = self.extract_features(self.satellite_img)

        # Извлекаем ключевые точки для отображения
        self.satellite_keypoints = self.extract_keypoints(self.satellite_img)
        
        self.display_photo_with_keypoints(self.satellite_keypoints)
        
        self.status_label.config(text=f"Спутник готов: {len(self.satellite_keypoints)} ключевых точек")
        print(f"Найдено {len(self.satellite_keypoints)} ключевых точек")
    
    def extract_features(self, img):
        """
        Извлекает признаки (ключевые точки + дескрипторы) для LightGlue.
        """

        img_np = np.array(img.convert('RGB'))
        tensor = transforms.ToTensor()(img_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.extractor.extract(tensor)

        return feats
    
    def extract_keypoints(self, img):
        """
        Извлекает только координаты ключевых точек.
        """

        img_np = np.array(img.convert('RGB'))
        tensor = transforms.ToTensor()(img_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.extractor.extract(tensor)

        feats = rbd(feats)

        return feats['keypoints'].cpu().numpy()
    
    def display_photo_with_keypoints(self, keypoints, max_points=100):
        """
        Отображает спутниковое фото с нанесёнными ключевыми точками.
        """

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
        """Загружает видео в плеер."""

        path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm")])
        if not path:
            return
        
        self.video_player.load(path)
        self.status_label.config(text="Видео загружено. Нажмите Play для запуска")
    
    def process_frame_background(self, frame_num):
        """
        Запускает фоновую обработку кадра для поиска совпадений.
        Вызывается из плеера каждый 15-й кадр.
        """

        if self.satellite_feats is None:
            return
        
        if frame_num == self.last_frame_processed:
            return
        
        self.last_frame_processed = frame_num
        
        # Запускаем обработку в отдельном потоке
        if self.matching_thread is None or not self.matching_thread.is_alive():
            self.matching_thread = threading.Thread(
                target=self.match_frame_in_background,
                args=(frame_num,),
                daemon=True
            )
            self.matching_thread.start()
    
    def ransac_filter(self, pts_sat, pts_frame, threshold=4.0, max_iter=2000):
        """
        Фильтрует ошибочные совпадения с помощью RANSAC.
        """

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
        """
        Проверяет, сгруппированы ли точки (не разбросаны по всему кадру).
        """

        if len(points) < 3:
            return False
        
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        mean_distance = np.mean(distances)
        
        return mean_distance < max_distance
    
    def match_frame_in_background(self, frame_num):
        """
        Основная функция обработки кадра в фоновом потоке.
        Выполняет извлечение признаков, сопоставление, фильтрацию.
        """

        # Получаем кадр через OpenCV 
        cap = cv2.VideoCapture(self.video_player.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return
        
        # Поиск лучшего угла 
        if self.rotation_search_enabled:
            video_points_raw, sat_points_raw, best_angle = self.find_best_rotation(frame)
            if len(video_points_raw) < 4:
                video_points_raw, sat_points_raw = self.match_frame(frame)
        else:
            video_points_raw, sat_points_raw = self.match_frame(frame)
        
        # Проверка минимального количества точек
        if len(video_points_raw) < 8:
            self.root.after(0, lambda: self.video_player.set_match_points([], [], None))
            return
        
        # RANSAC фильтрация
        sat_points_filt, video_points_filt, H = self.ransac_filter(
            sat_points_raw, video_points_raw, threshold=self.ransac_threshold
        )
        
        # Проверка кластеризации
        if len(video_points_filt) >= 4:
            is_clustered = self.check_points_clustering(video_points_filt, self.max_scatter_distance)
            
            if not is_clustered:
                print(f"Кадр {frame_num}: Точки слишком разбросаны - отклоняем")
                self.root.after(0, lambda: self.video_player.set_match_points([], [], None))
                return
            
            is_sat_clustered = self.check_points_clustering(sat_points_filt, self.max_scatter_distance * 2)
            
            if not is_sat_clustered:
                print(f"Кадр {frame_num}: Точки на спутнике слишком разбросаны - отклоняем")
                self.root.after(0, lambda: self.video_player.set_match_points([], [], None))
                return
            
            print(f"Кадр {frame_num}: ПРИНЯТО {len(video_points_filt)} совпадений")
            self.root.after(0, lambda: self.video_player.set_match_points(
                video_points_filt, sat_points_filt, H
            ))
        else:
            print(f"Кадр {frame_num}: Недостаточно совпадений после RANSAC: {len(video_points_filt)}")
            self.root.after(0, lambda: self.video_player.set_match_points([], [], None))
    
    def match_frame(self, frame):
        """
        Сопоставляет кадр видео со спутником (SuperPoint + LightGlue).
        """

        h, w = frame.shape[:2]
        
        # Обработка в оригинальном размере 
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
        
        # Извлечение признаков
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        tensor = transforms.ToTensor()(rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            frame_feats = self.extractor.extract(tensor)
        
        # Сопоставление со спутником
        with torch.no_grad():
            matches = self.matcher({"image0": self.satellite_feats, "image1": frame_feats})
        
        matches = rbd(matches)
        
        # Получение координат
        kpts_sat = self.satellite_feats["keypoints"][0].cpu().numpy()
        kpts_frame = frame_feats["keypoints"][0].cpu().numpy()
        matches_idx = matches["matches"].cpu().numpy()
        scores = matches["scores"].cpu().numpy()
        
        if len(matches_idx) < 4:
            return [], []
        
        # Фильтрация по уверенности
        conf_mask = scores > self.confidence_threshold
        matches_idx = matches_idx[conf_mask]
        
        if len(matches_idx) < 4:
            return [], []
        
        # Координаты совпавших точек
        sat_points = kpts_sat[matches_idx[:, 0]]
        frame_points = kpts_frame[matches_idx[:, 1]]
        
        # Масштабирование обратно к исходному размеру
        frame_points_original = frame_points * [scale_x, scale_y]
        
        return frame_points_original, sat_points
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()