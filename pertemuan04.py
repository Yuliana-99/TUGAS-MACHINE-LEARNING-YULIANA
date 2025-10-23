# LANGKAH 2 - COLLECTION
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Baca file csv
df = pd.read_csv("kelulusan_mahasiswa.csv")

# Info awal
print(df.info())
print(df.head())

# LANGKAH 3 - CLEANING
print("Missing values:\n", df.isnull().sum())
df = df.drop_duplicates()

# Visualisasi outlier
sns.boxplot(x=df['IPK'])
plt.title("Boxplot IPK")
plt.tight_layout()
plt.show()

# LANGKAH 4 - EDA
print(df.describe())

# Histogram distribusi IPK
sns.histplot(df['IPK'], bins=10, kde=True)
plt.title("Distribusi IPK")
plt.tight_layout()
plt.show()

# Scatterplot IPK vs Waktu Belajar
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.title("IPK vs Waktu Belajar (dengan Lulus)")
plt.tight_layout()
plt.show()

# Heatmap korelasi
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Korelasi Antar Fitur")
plt.tight_layout()
plt.show()

# LANGKAH 5 - FEATURE ENGINEERING
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

# Simpan dataset hasil preprocessing
df.to_csv("processed_kelulusan.csv", index=False)

# LANGKAH 6 - SPLITTING DATASET
X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Cek distribusi kelas
print("Distribusi kelas secara keseluruhan:")
print(y.value_counts())

# Cek apakah stratify bisa digunakan
if y.value_counts().min() > 1:
    # Split train & temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Cek apakah y_temp masih bisa stratify
    if y_temp.value_counts().min() > 1:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
    else:
        # Tidak bisa stratify, split biasa
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
else:
    # Tidak bisa stratify dari awal
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

# Final print
print("\nJumlah data per kelas (train):")
print(y_train.value_counts())
print("\nJumlah data per kelas (val):")
print(y_val.value_counts())
print("\nJumlah data per kelas (test):")
print(y_test.value_counts())

print("\nShapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)
