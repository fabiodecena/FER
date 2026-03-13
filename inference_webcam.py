import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EMOTION_COLORS = {
    "angry":    (0, 0, 255),
    "disgust":  (0, 128, 128),
    "fear":     (128, 0, 128),
    "happy":    (0, 255, 0),
    "neutral":  (200, 200, 200),
    "sad":      (255, 128, 0),
    "surprise": (0, 255, 255),
}


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


# ── Face detection (OpenCV Haar Cascade) ──────────────────────────
def setup_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Cannot load cascade: {cascade_path}")
    return detector


def detect_face(detector, frame_bgr) -> tuple[int, int, int, int] | None:
    """Ritorna (x, y, w, h) del volto più grande, o None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        return None

    # Prendi il volto più grande
    best = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = best

    # Espandi il box del 20%
    pad_x = int(w * 0.2)
    pad_y = int(h * 0.2)
    fh, fw = frame_bgr.shape[:2]
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(fw - x, w + 2 * pad_x)
    h = min(fh - y, h + 2 * pad_y)

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

    # ── Carica modello ────────────────────────────────────────────
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
    print(f"▶ Model: {ckpt['arch']}, classes: {classes}")
    print(f"▶ Screenshot trigger: any emotion >= {SCREENSHOT_THRESHOLD:.0%}")

    # ── Face detector ─────────────────────────────────────────────
    face_detector = setup_face_detector()
    smoother = PredictionSmoother(window=SMOOTHING_WINDOW)
    print("▶ Face detector: OpenCV Haar Cascade")

    # ── Webcam ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        popup_root.destroy()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("▶ Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Detect face ──────────────────────────────────────────
        face_box = detect_face(face_detector, frame)

        if face_box is not None:
            x, y, w, h = face_box
            face_crop_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)

            if face_crop_rgb.size == 0:
                continue

            # ── Inference ────────────────────────────────────────
            input_tensor = preprocess(face_crop_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = fnc.softmax(logits, dim=1).cpu().numpy()[0]

            # ── Smoothing ────────────────────────────────────────
            smoothed = smoother.update(probs)
            pred_idx = int(np.argmax(smoothed))
            pred_emotion = classes[pred_idx]
            pred_conf = smoothed[pred_idx]

            # ── Screenshot prompt ────────────────────────────────
            if pred_conf >= SCREENSHOT_THRESHOLD and not screenshot_prompt_shown:
                popup_root.update()
                save_image = messagebox.askyesno(
                    "Emotion detected",
                    f"Detected emotion: {pred_emotion} ({pred_conf:.0%}).\nDo you want to save this image?",
                    parent=popup_root,
                )

                if save_image:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = os.path.join(
                        SCREENSHOT_DIR,
                        f"{pred_emotion}_{int(pred_conf * 100)}_{timestamp}.png",
                    )
                    cv2.imwrite(screenshot_path, frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")

                screenshot_prompt_shown = True
            elif pred_conf < SCREENSHOT_THRESHOLD:
                screenshot_prompt_shown = False

            # ── Draw bounding box ────────────────────────────────
            color = EMOTION_COLORS.get(pred_emotion, (255, 255, 255))

            if pred_conf >= CONFIDENCE_THRESHOLD:
                label = f"{pred_emotion} {pred_conf:.0%}"
            else:
                label = "uncertain"
                color = (128, 128, 128)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # ── Label background ─────────────────────────────────
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, y - 30), (x + label_size[0] + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # ── Probability bars (lato destro) ───────────────────
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

        # ── Show ─────────────────────────────────────────────────
        cv2.imshow("FER - Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    popup_root.destroy()
    print("▶ Webcam closed.")


if __name__ == "__main__":
    main()