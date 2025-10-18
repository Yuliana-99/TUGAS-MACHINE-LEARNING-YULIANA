# LANGKAH 2 - COLLECTION
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Baca file csv
df = pd.read_csv("kelulusan_mahasiswa.csv")

# Cek info dan tampilkan 5 baris pertama
print(df.info())
print(df.head())

# LANGKAH 3 - CLEANING
# Cek missing values
print("Missing values:\n", df.isnull().sum())

# Hapus duplikat jika ada
df = df.drop_duplicates()

# Visualisasi boxplot untuk deteksi outlier
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['IPK'])
plt.show()

# LANGKAH 4 - EXPLOTARY DATA ANALYSIS ( EDA )
# Statistik deskriptif
print(df.describe())

# Histogram distribusi IPK
sns.histplot(df['IPK'], bins=10, kde=True)
plt.show()

# Scatterplot hubungan IPK vs Waktu Belajar, dengan hue=Lulus
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.show()

# Heatmap korelasi
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# LANGKAH 5 - FEATURE ENGINEERING
# Buat fitur turunan
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

# Simpan dataset baru
df.to_csv("processed_kelulusan.csv", index=False)

# LANGKAH 6 - SPLITTING DATASET
from sklearn.model_selection import train_test_split

# Pisahkan fitur (X) dan label (y)
X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Train (70%) dan sisanya 30% untuk val+test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Dari sisa 30%, bagi 50:50 -> masing-masing 15%
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("Train:", X_train.shape, "Validation:", X_val.shape, "Test:", X_test.shape)