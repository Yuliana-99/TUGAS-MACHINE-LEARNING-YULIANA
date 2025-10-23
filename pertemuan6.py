import pandas as pd
from sklearn.model_selection import train_test_split

# Membaca dataset
df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Cek distribusi kelas keseluruhan
class_counts = y.value_counts()
print("Distribusi Kelas (seluruh data):")
print(class_counts)

# Split: 70% train, 15% val, 15% test dengan stratify jika memungkinkan
if class_counts.min() > 1:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    if y_temp.value_counts().min() > 1:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
        )
    else:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42
        )
else:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

print("\nDistribusi Kelas di Train:")
print(y_train.value_counts())
print("\nDistribusi Kelas di Validation:")
print(y_val.value_counts())
print("\nDistribusi Kelas di Test:")
print(y_test.value_counts())

# ================= PIPELINE =================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

num_cols = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols),
], remainder="drop")

rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)

pipe = Pipeline([
    ("pre", pre),
    ("clf", rf)
])

pipe.fit(X_train, y_train)

y_val_pred = pipe.predict(X_val)
print("\nBaseline RF — F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# ================= CROSS-VALIDATION & GRID SEARCH =================
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

# Tentukan n_splits disesuaikan dengan kelas minoritas di train
min_class_count = y_train.value_counts().min()
n_splits = min(5, min_class_count)
print(f"\nJumlah fold untuk StratifiedKFold: {n_splits}")

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
print("CV F1-macro (train):", scores.mean(), "±", scores.std())

param = {
    "clf__max_depth": [None, 12, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}

gs = GridSearchCV(pipe, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print("\nBest params:", gs.best_params_)

best_model = gs.best_estimator_

y_val_best = best_model.predict(X_val)
print("\nBest RF — F1(val):", f1_score(y_val, y_val_best, average="macro"))

# ================= EVALUASI DI TEST SET =================
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

final_model = best_model

y_test_pred = final_model.predict(X_test)
print("\nF1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    try:
        print("ROC-AUC(test):", roc_auc_score(y_test, y_test_proba))
    except:
        pass

    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure(); plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (test)")
    plt.tight_layout(); plt.savefig("roc_test.png", dpi=120)

    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure(); plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve (test)")
    plt.tight_layout(); plt.savefig("pr_test.png", dpi=120)

# ================= FEATURE IMPORTANCE =================
try:
    import numpy as np
    importances = final_model.named_steps["clf"].feature_importances_
    fn = final_model.named_steps["pre"].get_feature_names_out()
    top = sorted(zip(fn, importances), key=lambda x: x[1], reverse=True)
    print("\nTop feature importance:")
    for name, val in top[:10]:
        print(f"{name}: {val:.4f}")
except Exception as e:
    print("Feature importance tidak tersedia:", e)

# ================= SIMPAN MODEL =================
import joblib
joblib.dump(final_model, "rf_model.pkl")
print("\nModel disimpan sebagai rf_model.pkl")

# ================= PREDIKSI DATA BARU =================
sample = pd.DataFrame([{
    "IPK": 3.4,
    "Jumlah_Absensi": 4,
    "Waktu_Belajar_Jam": 7,
    "Rasio_Absensi": 4/14,
    "IPK_x_Study": 3.4 * 7
}])
mdl = joblib.load("rf_model.pkl")
print("\nPrediksi:", int(mdl.predict(sample)[0]))
