"""
inference_webcam.py - Marketing Research Edition (v2.0)

Key Features:
- Ghost UI: User sees a clean mirror; Department receives analyzed frames.
- Rate-Limiting: 5-second cooldown between captures to ensure high-quality data.
- Cloud-Ready: Background hooks for S3 and GitHub LFS versioning.
- Robust Logging: Immediate CSV flushing for real-time analytics.
"""

import os
import csv
import time
import torch
import cv2
import boto3
import numpy as np
import torch.nn.functional as fnc
from datetime import datetime
from torchvision import transforms
from model import build_model
import sys

import sys
import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- UPDATE YOUR PATHS USING THE FUNCTION ---
CHECKPOINT = resource_path("Merged/checkpoints/merged_best_20260305_162825.pt")
CASCADE_FILE = resource_path("haarcascade_frontalface_default.xml")
SCREENSHOT_DIR = "screenshots"
FEEDBACK_LOG = "research_analytics_log.csv"

# Thresholds & Timing
CONF_THRESHOLD = 0.70      # Quality trigger
COOLDOWN_SECONDS = 3.0     # Prevent rapid-fire saving
IMG_SIZE = 224

# AWS S3 Credentials (Defaults to DUMMY to prevent errors)
S3_BUCKET = os.getenv("S3_BUCKET", "marketing-sentiment-data")
AWS_KEY = os.getenv("AWS_KEY", "DUMMY_KEY")
AWS_SECRET = os.getenv("AWS_SECRET", "DUMMY_SECRET")

EMOTION_COLORS = {
    "angry": (0, 0, 255), "disgust": (0, 128, 128), "fear": (128, 0, 128),
    "happy": (0, 255, 0), "neutral": (200, 200, 200), "sad": (255, 128, 0),
    "surprise": (0, 255, 255)
}

# ── Helper: Cloud Sync ──────────────────────────────────────────
def sync_to_cloud(file_path):
    """Silent upload to S3. Skips if using DUMMY credentials."""
    if "DUMMY" not in AWS_KEY:
        try:
            s3 = boto3.client('s3', aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET)
            s3.upload_file(file_path, S3_BUCKET, file_path)
            print(f"☁️ Cloud Sync: {file_path} pushed to S3.")
        except Exception as e:
            print(f"⚠️ S3 Sync Skipped: {e}")

# ── Helper: Robust Logging ──────────────────────────────────────
def log_research_data(img_filename, emotion, confidence):
    """
    Logs metadata to CSV with a placeholder for Human Audit.
    Ensures the file is flushed to disk for immediate S3/LFS sync.
    """
    file_exists = os.path.isfile(FEEDBACK_LOG)
    with open(FEEDBACK_LOG, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Professional Header with Audit Column
        if not file_exists:
            writer.writerow([
                'timestamp',
                'image_path',
                'ai_predicted_emotion',
                'ai_confidence',
                'true_label'  # The Audit Column
            ])

        # Log the data with "pending_review" as the placeholder
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            img_filename,
            emotion,
            f"{confidence:.4f}",
            "pending_review"  # Marketing Analyst will change this
        ])

    # Sync the log file to the cloud
    sync_to_cloud(FEEDBACK_LOG)

# ── Preprocessing ────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Main Application ─────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # Initialization
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model = build_model(num_classes=ckpt["num_classes"], backbone=ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    classes = ckpt["classes"]
    detector = cv2.CascadeClassifier(CASCADE_FILE)
    if detector.empty():
        print(f"❌ Error: Cannot load face detector at {CASCADE_FILE}")
    cap = cv2.VideoCapture(0)

    last_capture_time = 0
    print("▶ Research Probe Active. User UI: 'Ghost Mirror'.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Stream Management
        user_display = frame.copy()    # Clean Feed
        research_data = frame.copy()   # Analysis Feed

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            # Inference Logic
            face_roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(face_roi).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = fnc.softmax(logits, dim=1).cpu().numpy()[0]

            pred_idx = np.argmax(probs)
            emotion = classes[pred_idx]
            conf = probs[pred_idx]

            # Analysis Overlay (Internal Only)
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            cv2.rectangle(research_data, (x, y), (x+w, y+h), color, 3)
            cv2.putText(research_data, f"{emotion.upper()} ({conf:.2f})", (x, y-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # ── Capture with Cooldown ──
            current_time = time.time()
            if conf >= CONF_THRESHOLD and (current_time - last_capture_time) > COOLDOWN_SECONDS:
                timestamp = datetime.now().strftime("%H%M%S")
                filename = f"{SCREENSHOT_DIR}/{emotion}_{timestamp}.png"

                # 1. Save Image (Research Frame)
                cv2.imwrite(filename, research_data)

                # 2. Log Data & Sync
                log_research_data(filename, emotion, conf)
                sync_to_cloud(filename)

                last_capture_time = current_time
                print(f"📸 Captured: {emotion} at {conf:.2f}. Cooldown engaged.")

        # Display the Ghost Mirror
        cv2.imshow("Marketing Study - Participant Feed", user_display)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()