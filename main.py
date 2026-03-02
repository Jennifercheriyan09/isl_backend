import os
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ISL Recognition API")

# --- CONFIGURATION FROM YOUR TRAIN SCRIPT ---
SEQUENCE_LENGTH = 30
HAND_START = 303  # 99 (pose) + 204 (face)
RAW_FEATURES = 429
FINAL_FEATURES = 552

# --- LOAD ASSETS ---
# These must be in the same folder as main.py on Render
model = tf.keras.models.load_model('best_isl_model.keras')
scaler = joblib.load('isl_scaler.pkl')
label_classes = np.load('label_classes.npy', allow_pickle=True)

class PredictionRequest(BaseModel):
    # Expects a list of 30 frames, each containing 429 float landmarks
    sequence: list[list[float]]

def add_velocity_compute(sequence_np):
    """Exact logic from your training script"""
    hand = sequence_np[:, HAND_START:]  # (30, 126)
    vel = np.zeros_like(hand)
    vel[1:] = hand[1:] - hand[:-1]
    return np.concatenate([sequence_np, vel], axis=1) # (30, 552)

@app.get("/")
def home():
    return {"status": "online", "classes": label_classes.tolist()}

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # 1. Convert to numpy array
        X_raw = np.array(request.sequence, dtype=np.float32)
        
        if X_raw.shape != (SEQUENCE_LENGTH, RAW_FEATURES):
            raise ValueError(f"Expected (30, 429), got {X_raw.shape}")

        # 2. Add Velocity (429 -> 552)
        X_with_vel = add_velocity_compute(X_raw)

        # 3. Scale (RobustScaler expects flattened input, then reshape back)
        X_scaled = scaler.transform(X_with_vel.reshape(-1, FINAL_FEATURES))
        X_final = X_scaled.reshape(1, SEQUENCE_LENGTH, FINAL_FEATURES)

        # 4. Inference
        preds = model.predict(X_final, verbose=0)[0]
        idx = np.argmax(preds)
        
        return {
            "prediction": str(label_classes[idx]),
            "confidence": float(preds[idx]),
            "top_3": {
                str(label_classes[i]): float(preds[i]) 
                for i in np.argsort(preds)[-3:][::-1]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))