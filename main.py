import os
import cv2
import joblib
import tempfile
import numpy as np
import tensorflow as tf
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI(title="ISL Recognition API")
print("THIS IS ISL BACKEND")

# ── CONFIG — matched exactly to training notebook ─────────────────────────────
FACE_KEY_INDICES = [
    0, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
    146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311,
    312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
    46, 53, 52, 65, 55, 70, 276, 283, 282, 295, 285,
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    362, 382, 381, 380, 374, 373, 390, 249, 263,
]

SEQUENCE_LENGTH = 30
HAND_START      = 99 + len(FACE_KEY_INDICES) * 3  # = 300
RAW_FEATURES    = 33*3 + len(FACE_KEY_INDICES)*3 + 21*3 + 21*3  # = 426
FINAL_FEATURES  = RAW_FEATURES + 21*3 + 21*3                     # = 552

# Training always resized to 640x480 (4:3) before MediaPipe
FRAME_W, FRAME_H = 640, 480

# ── LOAD MODEL ASSETS ─────────────────────────────────────────────────────────
model         = tf.keras.models.load_model('best_isl_model.keras')
scaler        = joblib.load('isl_scaler.pkl')
label_classes = np.load('label_classes.npy', allow_pickle=True)

print(f"Loaded {len(label_classes)} classes: {label_classes.tolist()}")

# ── MEDIAPIPE ─────────────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic


# ── FRAME PREPROCESSING ───────────────────────────────────────────────────────
def crop_to_training_ratio(frame: np.ndarray) -> np.ndarray:
    """
    Center-crop frame to 4:3 ratio then resize to 640x480.
    Avoids distortion when input video has a different aspect ratio.
    """
    h, w          = frame.shape[:2]
    target_ratio  = FRAME_W / FRAME_H  # 4:3 = 1.333
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left  = (w - new_w) // 2
        frame = frame[:, left: left + new_w]
    elif current_ratio < target_ratio:
        new_h = int(w / target_ratio)
        top   = (h - new_h) // 2
        frame = frame[top: top + new_h, :]

    return cv2.resize(frame, (FRAME_W, FRAME_H))


# ── LANDMARK EXTRACTION ───────────────────────────────────────────────────────
def extract_landmarks(results) -> np.ndarray:
    # Pose
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.pose_landmarks.landmark])
        pose = (pose - pose[0]).flatten()
    else:
        pose = np.zeros(33 * 3)

    # Face (key landmarks only)
    if results.face_landmarks:
        all_face = np.array([[lm.x, lm.y, lm.z]
                              for lm in results.face_landmarks.landmark])
        all_face = all_face - all_face[1]
        face     = all_face[FACE_KEY_INDICES].flatten()
    else:
        face = np.zeros(len(FACE_KEY_INDICES) * 3)

    # Left hand
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                        for lm in results.left_hand_landmarks.landmark])
        lh = (lh - lh[0]).flatten()
    else:
        lh = np.zeros(21 * 3)

    # Right hand
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                        for lm in results.right_hand_landmarks.landmark])
        rh = (rh - rh[0]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, face, lh, rh])  # (426,)


# ── VELOCITY ──────────────────────────────────────────────────────────────────
def add_velocity(sequence_np: np.ndarray) -> np.ndarray:
    """Appends hand velocity features. (30, 426) → (30, 552)"""
    hand    = sequence_np[:, HAND_START:]
    vel     = np.zeros_like(hand)
    vel[1:] = hand[1:] - hand[:-1]
    return np.concatenate([sequence_np, vel], axis=1)


# ── ACTIVE FRAME DETECTION ────────────────────────────────────────────────────
def is_hands_active(frame_landmarks: np.ndarray) -> bool:
    lh = frame_landmarks[HAND_START: HAND_START + 63]
    rh = frame_landmarks[HAND_START + 63: HAND_START + 126]
    return not (np.all(lh == 0) and np.all(rh == 0))


# ── VIDEO EXTRACTION ──────────────────────────────────────────────────────────
def extract_sequence_from_video(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        raise ValueError("Video has no frames")

    all_landmarks = []

    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        ) as holistic:
            for _ in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    all_landmarks.append(np.zeros(RAW_FEATURES))
                    continue

                frame   = crop_to_training_ratio(frame)
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb     = np.ascontiguousarray(rgb, dtype=np.uint8)
                results = holistic.process(rgb)
                all_landmarks.append(extract_landmarks(results))
    finally:
        cap.release()

    active_frames = [f for f in all_landmarks if is_hands_active(f)]
    print(f"[DEBUG] Total: {len(all_landmarks)}, Active: {len(active_frames)}")

    if len(active_frames) >= SEQUENCE_LENGTH:
        indices  = np.linspace(0, len(active_frames) - 1, SEQUENCE_LENGTH, dtype=int)
        sequence = np.array(active_frames)[indices]
    elif len(active_frames) > 0:
        print(f"[WARN] Only {len(active_frames)} active frames, padding to {SEQUENCE_LENGTH}")
        pad      = [active_frames[-1]] * (SEQUENCE_LENGTH - len(active_frames))
        sequence = np.array(active_frames + pad)
    else:
        print("[WARN] No active hands detected, falling back to full video")
        indices  = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
        sequence = np.array(all_landmarks)[indices]

    return sequence.astype(np.float32)  # (30, 426)


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "status":         "online",
        "classes":        label_classes.tolist(),
        "expected_input": "POST /predict with video file (mp4/mov/avi)"
    }


# ── ONLY CHANGE IS HERE — landmarks added to response ─────────────────────────
@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    suffix = os.path.splitext(video.filename)[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        # 1. Extract (30, 426) landmark sequence
        X_raw = extract_sequence_from_video(tmp_path)

        if X_raw.shape != (SEQUENCE_LENGTH, RAW_FEATURES):
            raise ValueError(
                f"Bad sequence shape: {X_raw.shape}, "
                f"expected ({SEQUENCE_LENGTH}, {RAW_FEATURES})"
            )

        # 2. Add velocity → (30, 552)
        X_vel = add_velocity(X_raw)

        # 3. Scale
        X_scaled = scaler.transform(
            X_vel.reshape(-1, FINAL_FEATURES)
        ).reshape(1, SEQUENCE_LENGTH, FINAL_FEATURES)

        # 4. Predict
        preds    = model.predict(X_scaled, verbose=0)[0]
        idx      = int(np.argmax(preds))
        top5_idx = np.argsort(preds)[::-1][:5]

        print(f"[DEBUG] Predicted: {label_classes[idx]} ({preds[idx]:.3f})")

        # ── CHANGE: added landmarks to response ──────────────────────────────
        # X_raw is (30, 426) raw unscaled — adaptive learning will re-apply
        # velocity + scaling itself to match the training pipeline exactly
        return {
            "label":      str(label_classes[idx]),
            "confidence": float(preds[idx]),
            "landmarks":  X_raw.tolist(),   # ← NEW: (30, 426) raw landmarks
            "top_5": [
                {
                    "label":      str(label_classes[i]),
                    "confidence": float(preds[i])
                }
                for i in top5_idx
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass