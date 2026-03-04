import joblib
import numpy as np

scaler = joblib.load('isl_scaler.pkl')

print(f"Type: {type(scaler)}")
print(f"Center (median) — min: {scaler.center_.min():.4f}, max: {scaler.center_.max():.4f}")
print(f"Scale (IQR)     — min: {scaler.scale_.min():.4f}, max: {scaler.scale_.max():.4f}")

# RobustScaler formula: (x - median) / IQR
# So original training data range was roughly:
lower = scaler.center_ - 1.5 * scaler.scale_
upper = scaler.center_ + 1.5 * scaler.scale_
print(f"\nEstimated training data range (median ± 1.5*IQR):")
print(f"  Lower bound: {lower.min():.4f}")
print(f"  Upper bound: {upper.max():.4f}")

# Also check what your phone value of 2.193 maps to after scaling
# RobustScaler: scaled = (x - median) / IQR
# If x=2.193 and IQR is tiny, scaled explodes
worst_iqr = scaler.scale_.min()
worst_center = scaler.center_.max()
print(f"\nWorst case: smallest IQR is {worst_iqr:.6f}")
print(f"A value of 2.193 on that feature scales to: {(2.193 - worst_center) / worst_iqr:.2f}")