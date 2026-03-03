import os
import cv2
import joblib
import tempfile
import numpy as np
import tensorflow as tf
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI(title="ISL Recognition API")

# ── EXACT SAME CONFIG AS YOUR REALTIME SCRIPT ─────────────────────────────────
FACE_KEY_INDICES = [
    0, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
    146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311,
    312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
    46, 53, 52, 65, 55, 70, 276, 283, 282, 295, 285,
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    362, 382, 381, 380, 374, 373, 390, 249, 263,
]

SEQUENCE_LENGTH = 30
HAND_START      = 99 + len(FACE_KEY_INDICES) * 3  # 303
RAW_FEATURES    = 33*3 + len(FACE_KEY_INDICES)*3 + 21*3 + 21*3  # 426
FINAL_FEATURES  = RAW_FEATURES + 21*3 + 21*3                     # 552

# ── LOAD MODEL ASSETS ─────────────────────────────────────────────────────────
model         = tf.keras.models.load_model('best_isl_model.keras')
scaler        = joblib.load('isl_scaler.pkl')
label_classes = np.load('label_classes.npy', allow_pickle=True)

# ── MEDIAPIPE ─────────────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic


# ── LANDMARK EXTRACTION (identical to your realtime script) ───────────────────
def extract_normalized_landmarks(results):
    # Pose: 33 joints × 3 = 99
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.pose_landmarks.landmark])
        pose = (pose - pose[0]).flatten()
    else:
        pose = np.zeros(33 * 3)

    # Face: key indices only
    if results.face_landmarks:
        all_face = np.array([[lm.x, lm.y, lm.z]
                              for lm in results.face_landmarks.landmark])
        all_face = all_face - all_face[1]
        face = all_face[FACE_KEY_INDICES].flatten()
    else:
        face = np.zeros(len(FACE_KEY_INDICES) * 3)

    # Left hand: 21 × 3 = 63
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                        for lm in results.left_hand_landmarks.landmark])
        lh = (lh - lh[0]).flatten()
    else:
        lh = np.zeros(21 * 3)

    # Right hand: 21 × 3 = 63
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                        for rh in results.right_hand_landmarks.landmark])
        rh = (rh - rh[0]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, face, lh, rh])  # (426,)


def add_velocity(sequence_np):
    hand = sequence_np[:, HAND_START:]       # (30, 126)
    vel  = np.zeros_like(hand)
    vel[1:] = hand[1:] - hand[:-1]
    return np.concatenate([sequence_np, vel], axis=1)  # (30, 552)


def extract_sequence_from_video(video_path: str) -> np.ndarray:
    """
    Reads video, samples exactly SEQUENCE_LENGTH frames evenly,
    runs MediaPipe on each, returns (30, 426) array.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        raise ValueError("Video has no frames")

    # Pick SEQUENCE_LENGTH evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)

    sequence = []
    with mp_holistic.Holistic(
        static_image_mode=True,          # treat each frame independently
        min_detection_confidence=0.3,
        model_complexity=1
    ) as holistic:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                sequence.append(np.zeros(RAW_FEATURES))
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            sequence.append(extract_normalized_landmarks(results))

    cap.release()
    return np.array(sequence, dtype=np.float32)  # (30, 426)


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "status": "online",
        "classes": label_classes.tolist(),
        "expected_input": "POST /predict with video file (mp4/mov/avi)"
    }


@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    # 1. Save uploaded video to a temp file
    suffix = os.path.splitext(video.filename)[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        # 2. Extract (30, 426) landmark sequence from video
        X_raw = extract_sequence_from_video(tmp_path)

        if X_raw.shape != (SEQUENCE_LENGTH, RAW_FEATURES):
            raise ValueError(f"Bad sequence shape: {X_raw.shape}, expected ({SEQUENCE_LENGTH}, {RAW_FEATURES})")

        # 3. Add velocity → (30, 552)
        X_vel = add_velocity(X_raw)

        # 4. Scale
        X_scaled = scaler.transform(
            X_vel.reshape(-1, FINAL_FEATURES)
        ).reshape(1, SEQUENCE_LENGTH, FINAL_FEATURES)

        # 5. Predict
        preds = model.predict(X_scaled, verbose=0)[0]
        idx   = int(np.argmax(preds))

        top5_idx = np.argsort(preds)[::-1][:5]

        return {
            "label":      str(label_classes[idx]),
            "confidence": float(preds[idx]),
            "top_5": [
                {"label": str(label_classes[i]), "confidence": float(preds[i])}
                for i in top5_idx
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        os.unlink(tmp_path)  # always clean up temp file