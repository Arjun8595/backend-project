import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

print("🚀 Creating fresh model...")

# Simple dummy data
X = np.array([
    [0,0,0,80,0],
    [1,1,1,40,3],
    [2,2,0,50,2],
    [0,1,0,60,1],
    [1,0,1,90,3]
])

y = np.array([1,2,2,1,3])

model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("MODEL FIXED & READY")