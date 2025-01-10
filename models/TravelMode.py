import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import sounddevice as sd
import os
import soundfile as sf

class DepthDetection:
    def __init__(self):
        # Initialize YOLO model
        self.yolo_model = YOLO('yolov8n.pt')  # Use YOLOv8 nano model

        # Depth estimation model (MiDaS small for efficiency)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.to(self.device)
        self.midas.eval()

        # MiDaS input transform
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

        # Camera parameters
        self.CAMERA_FOV = 70  # Field of view in degrees
        self.FRAME_WIDTH = 640
        self.FRAME_HEIGHT = 480

        # Global variables to track detected objects
        self.previous_objects = set()

    def fractional_delay(self, audio, delay, sample_rate):
        delay_samples = delay * sample_rate
        integer_delay = int(np.floor(delay_samples))
        fractional = delay_samples - integer_delay
        padded_audio = np.pad(audio, (integer_delay + 2, 0), mode='constant')
        delayed_audio = (1 - fractional) * padded_audio[:-integer_delay - 2] + fractional * padded_audio[1:-integer_delay - 1]
        return delayed_audio[:len(audio)]

    def spherical_to_cartesian(self, r, theta, gamma):
        theta = np.radians(theta)
        gamma = np.radians(gamma)
        x = r * np.sin(gamma) * np.cos(theta)
        y = r * np.sin(gamma) * np.sin(theta)
        z = r * np.cos(gamma)
        return x, y, z

    def calculate_attenuation(self, r, absorption_coefficient=0.01):
        attenuation = 1 / (r ** 2)
        attenuation *= np.exp(-absorption_coefficient * r)
        return attenuation

    def calculate_ild(self, theta, gamma, max_ild_db=15):
        ild = max_ild_db * np.cos(np.radians(theta))
        ild_left_db = -ild / 2
        ild_right_db = ild / 2
        return ild_left_db, ild_right_db

    def apply_spherical_spatialization(self, audio, r, theta, gamma, sample_rate):
        attenuation = self.calculate_attenuation(r)
        max_itd = 0.0008  # 800 microseconds
        itd = max_itd * np.sin(np.radians(theta)) * np.cos(np.radians(gamma))
        ild_left_db, ild_right_db = self.calculate_ild(theta, gamma)
        ild_left = 10 ** (ild_left_db / 20.0)
        ild_right = 10 ** (ild_right_db / 20.0)
        left = audio * ild_left * attenuation
        right = audio * ild_right * attenuation
        left = self.fractional_delay(left, max_itd + itd, sample_rate)
        right = self.fractional_delay(right, max_itd - itd, sample_rate)
        return np.vstack((left, right)).T

    def spatialize_with_spherical_coords(self, text, r, theta, gamma, sample_rate, duration):
        tts = gTTS(text)
        tts.save("temp_tts.mp3")
        audio, file_sr = sf.read("temp_tts.mp3")
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        desired_length = int(sample_rate * duration)
        if len(audio) < desired_length:
            audio = np.pad(audio, (0, desired_length - len(audio)), mode='constant')
        else:
            audio = audio[:desired_length]
        spatial_audio = self.apply_spherical_spatialization(audio, r, theta, gamma, sample_rate)
        spatial_audio = spatial_audio / np.max(np.abs(spatial_audio))
        print(f"Playing from: r={r}, θ={theta}, γ={gamma}")
        sd.play(spatial_audio, samplerate=sample_rate)
        sd.wait()
        os.remove("temp_tts.mp3")

    def process_frame(self, frame):
        # Preprocess frame for depth estimation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(frame_rgb).to(self.device)
        with torch.no_grad():
            depth = self.midas(input_batch)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(self.FRAME_HEIGHT, self.FRAME_WIDTH),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
        depth = cv2.normalize(depth, None, 0, 1, norm_type=cv2.NORM_MINMAX)

        # YOLO object detection
        results = self.yolo_model(frame)
        detections = results[0].boxes

        # Process detected objects
        current_objects = set()
        for i, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            confidence = detections.conf[i].item()
            class_id = int(detections.cls[i].item())
            obj_name = self.yolo_model.names[class_id]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            obj_depth = 1 + depth[center_y, center_x]
            pixel_offset = center_x - (self.FRAME_WIDTH // 2)
            angle = (pixel_offset / (self.FRAME_WIDTH // 2)) * (self.CAMERA_FOV / 2)
            current_objects.add((obj_name, obj_depth, angle))

        # Identify new objects
        new_objects = current_objects - self.previous_objects
        self.previous_objects = current_objects

        # Spatialize new objects
        for obj in new_objects:
            self.spatialize_with_spherical_coords(f"{obj[0]} detected at {obj[1]:.2f} meters", obj[1], obj[2], 0, 44100, 1.5)

        return frame
