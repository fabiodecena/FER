import os
import csv
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import torch
import torch.nn.functional as fnc
import numpy as np
from torchvision import transforms
from model import build_model

# ── Config ────────────────────────────────────────────────────────
CHECKPOINT = "Merged/checkpoints/merged_best_20260305_162825.pt"
CONFIDENCE_THRESHOLD = 0.35
SMOOTHING_WINDOW = 5
IMG_SIZE = 224

SCREENSHOT_THRESHOLD = 0.65
SCREENSHOT_DIR = "screenshots"
FEEDBACK_LOG = "user_feedback_log.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 128),
    "fear": (128, 0, 128),
    "happy": (0, 255, 0),
    "neutral": (200, 200, 200),
    "sad": (255, 128, 0),
    "surprise": (0, 255, 255),
}


# ── Logging Setup ────────────────────────────────────────────────
def log_feedback(filename, timestamp, pred_class, feedback_yes_no, true_label):
    file_exists = os.path.isfile(FEEDBACK_LOG)
    with open(FEEDBACK_LOG, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['filename', 'timestamp', 'predicted_class', 'user_feedback', 'true_label'])
        writer.writerow([filename, timestamp, pred_class, feedback_yes_no, true_label])


# ── Preprocessing ─────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Smoothing buffer ─────────────────────────────────────────────
class PredictionSmoother:
    def __init__(self, window: int = 5):
        self.window = window
        self.buffer = []

    def update(self, probs: np.ndarray) -> np.ndarray:
        self.buffer.append(probs)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        return np.mean(self.buffer, axis=0)

    def reset(self):
        self.buffer.clear()


# ── Face detection ───────────────────────────────────────────────
def setup_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Cannot load cascade: {cascade_path}")
    return detector


def detect_face(detector, frame_bgr) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0: return None
    best = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = best
    pad_x, pad_y = int(w * 0.2), int(h * 0.2)
    fh, fw = frame_bgr.shape[:2]
    x, y = max(0, x - pad_x), max(0, y - pad_y)
    w, h = min(fw - x, w + 2 * pad_x), min(fh - y, h + 2 * pad_y)
    return x, y, w, h


# ── Main loop ─────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Device: {device}")

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    screenshot_prompt_shown = False

    popup_root = tk.Tk()
    popup_root.withdraw()
    popup_root.attributes("-topmost", True)

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model = build_model(
        num_classes=ckpt["num_classes"],
        backbone=ckpt["arch"],
        pretrained=False,
        hidden_dim=0,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    classes = ckpt["classes"]
    face_detector = setup_face_detector()
    smoother = PredictionSmoother(window=SMOOTHING_WINDOW)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        popup_root.destroy()
        return

    print("▶ Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        face_box = detect_face(face_detector, frame)

        if face_box is not None:
            x, y, w, h = face_box
            face_crop_rgb = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
            if face_crop_rgb.size == 0: continue

            input_tensor = preprocess(face_crop_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = fnc.softmax(logits, dim=1).cpu().numpy()[0]

            smoothed = smoother.update(probs)
            pred_idx = int(np.argmax(smoothed))
            pred_emotion = classes[pred_idx]
            pred_conf = smoothed[pred_idx]

            # ── Feedback Logic ───────────────────────────────────
            if pred_conf >= SCREENSHOT_THRESHOLD and not screenshot_prompt_shown:
                popup_root.update()
                save_image = messagebox.askyesno(
                    "Detection Prompt",
                    f"Detected {pred_emotion} ({pred_conf:.0%}). Save image?",
                    parent=popup_root
                )

                if save_image:
                    now = datetime.now()
                    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    file_timestamp = now.strftime("%Y%m%d_%H%M%S")
                    filename = f"{pred_emotion}_{file_timestamp}.png"

                    cv2.imwrite(os.path.join(SCREENSHOT_DIR, filename), frame)

                    is_correct = messagebox.askyesno(
                        "Verify Accuracy",
                        f"Is '{pred_emotion}' correct?",
                        parent=popup_root
                    )

                    feedback_val = "yes" if is_correct else "no"
                    true_label = pred_emotion

                    if not is_correct:
                        true_label = simpledialog.askstring(
                            "Manual Label",
                            f"Correct emotion ({', '.join(classes)}):",
                            parent=popup_root
                        )
                        if not true_label: true_label = "unknown"

                    log_feedback(filename, timestamp_str, pred_emotion, feedback_val, true_label)

                screenshot_prompt_shown = True
            elif pred_conf < SCREENSHOT_THRESHOLD:
                screenshot_prompt_shown = False

            # ── Drawing: Bounding Box & Label ────────────────────
            color = EMOTION_COLORS.get(pred_emotion, (255, 255, 255))
            if pred_conf >= CONFIDENCE_THRESHOLD:
                label = f"{pred_emotion} {pred_conf:.0%}"
            else:
                label = "uncertain"
                color = (128, 128, 128)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, y - 30), (x + label_size[0] + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # ── Drawing: Probability bars (Right Side) ────────────
            bar_x = frame.shape[1] - 220
            bar_y_start = 30
            for i, (cls, prob) in enumerate(zip(classes, smoothed)):
                bar_y = bar_y_start + i * 30
                bar_w = int(prob * 180)
                c = EMOTION_COLORS.get(cls, (200, 200, 200))

                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), c, -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 180, bar_y + 20), (100, 100, 100), 1)
                cv2.putText(frame, f"{cls[:3]} {prob:.0%}", (bar_x - 70, bar_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        else:
            smoother.reset()
            screenshot_prompt_shown = False
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("FER - Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()
    popup_root.destroy()


if __name__ == "__main__":
    main()