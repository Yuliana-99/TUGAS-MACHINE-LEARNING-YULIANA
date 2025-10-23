import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Cek distribusi kelas
print("Distribusi kelas sebelum split:")
print(y.value_counts())

# Standardisasi fitur
sc = StandardScaler()

# Jangan fit_transform sekaligus sebelum split, 
# karena bisa menyebabkan data leak, jadi split dulu baru fit transform di train dan transform di val/test
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Cek distribusi kelas di temp
print("Distribusi kelas di temp (val+test):")
print(y_temp.value_counts())

# Split val dan test
# Cek apakah stratify masih bisa dilakukan tanpa error (minimal 2 data per kelas)
if y_temp.value_counts().min() > 1:
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
else:
    # Kalau stratify error, split biasa tanpa stratify
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.5, random_state=42)

print("Distribusi kelas di val:")
print(y_val.value_counts())
print("Distribusi kelas di test:")
print(y_test.value_counts())

# Fit scaler di data train saja
sc.fit(X_train_raw)
X_train = sc.transform(X_train_raw)
X_val = sc.transform(X_val_raw)
X_test = sc.transform(X_test_raw)

print(X_train.shape, X_val.shape, X_test.shape)

# =================== MODEL DEFINITION ===================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # binary classification
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", "AUC"]
)

model.summary()

# =================== EARLY STOPPING CALLBACK ===================
es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# =================== TRAINING ===================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# =================== EVALUASI DI TEST SET ===================
from sklearn.metrics import classification_report, confusion_matrix

loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f} | Test AUC: {auc:.4f}")

y_proba = model.predict(X_test).ravel()
y_pred = (y_proba >= 0.5).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=3))

# =================== LEARNING CURVE PLOTTING ===================
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Learning Curve")
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=120)
plt.show()
