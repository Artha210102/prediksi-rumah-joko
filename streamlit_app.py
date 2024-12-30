#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression
# - Kali ini Ucup mencoba untuk membantu temenya yaitu Joko untuk memprediksi harga rumah untuknya.
# - Joko senduri tinggal di US tepatnya di King County dan sekarang sedang mencari rumah karena dia baru saja menikaj.
# - Data diambil dari kaggle dengan sedikit modifikasi.
# - Joko sendiri ingin membeli rumah dengan jumlah kamar tidur itu 3, jumlah kamar mandinya itu 2, luas rumahnya itu 1800sqft, grade rumahnya 7 dan tahun pembangunanya pada tahun 1990.
# - Yuk bantu Ucup membangun model machine learning untuk membantu joko!

# - Langkah Pengerjaan hampir sama dengan yang Simple Linear Regression hanya saja Multivariate Linear Regression memiliki lebih > 1 independent variable (x)

# ### Load library

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# ### Load datasets

# In[2]:


#Nama dataframe kita adalah df yang berisi data dari kc_house_data.csv.
#Features yang digunakan adalah 'bedrooms','bathrooms','sqft_living','grade','price' dan 'yr_built'

df = pd.read_csv('/content/kc_house_data.csv', usecols=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'price', 'yr_built'])


# ### Sneak peak data

# In[3]:


#Melihat 5 baris teratas dari data
#Independent variabel(x) adalah bedrooms, bathrooms, sqft_living, grade, yr_built
#Dependent variabel(y) adalah price
df.head()


# - Penjelasan setiap kolom:
#     1. bedrooms = Jumlah kamar tidur
#     2. bathrooms = Jumlah kamar mandi
#     3. sqft_living = Luas rumah dalam satuan sqft
#     4. grade = Grading system dari pemerintah King County US
#     5. yr_built = Tahun dimana rumah dibangun
#     6. price = Harga dari rumah (US$)

# In[4]:


#Mengetahui jumlah kolom dan baris dari data
#Data kita mempunya 6 kolom (features) dengan 21613 baris
df.shape


# In[5]:


#Melihat informasi data kita mulai dari jumlah data, tipe data, memory yang digunakan dll.
#Dapat dilihat bahwa seluruh data sudah di dalam bentuk numerik
df.info()


# In[6]:


#Melihat statistical description dari data mulai dari mean, kuartil, standard deviation dll
df.describe()


# - Pada feature bathrooms terdapat nilai pecahan, aneh kan yak kalo ada nilai jumlah kamar mandi pecahan gitu. Maka kita ubah dulu jenis data yang semula float menjadi int.
# - Pada feature bedrooms terdapat nilai 33, ini sangat aneh karena masak rumah ada yang punya kamar 33 apalagi ini rumah pribadi. jadi kemungkinan itu typo dan akan saya ganti menjadi 3

# In[7]:


#Mrubah tipe data dari bathrooms yang semula float menjadi int
df['bathrooms'] = df['bathrooms'].astype('int')


# In[8]:


#Mengganti nilai 33 menjadi 3
df['bedrooms'] = df['bedrooms'].replace(33,3)


# ### Handling Missing Values

# In[9]:


#Mencari dan menangani missing values
#Ternyata data kita tidak ada missing values
df.isnull().sum()


# ### Exploratory Data Analysis (EDA)

# In[10]:


df.head()


# In[11]:


#Univariate analysis bedrooms
#Melihat distribusi dari bedrooms
f = plt.figure(figsize=(12,4))

f.add_subplot(1,2,1)
sns.countplot(df['bedrooms'])

f.add_subplot(1,2,2)
plt.boxplot(df['bedrooms'])
plt.show()


# - Dapat dilihat bahwa sebagian besar jumlah kamar tidur itu di angka 3 dan 4.
# - Data memiliki banyak outliers.

# In[13]:


#Univariate analysis bathrooms
#Melihat distribusi dari bathrooms
f = plt.figure(figsize=(12,4))

f.add_subplot(1,2,1)
sns.countplot(df['bathrooms'])

f.add_subplot(1,2,2)
plt.boxplot(df['bathrooms'])
plt.show()


# - Jumlah kamar mandi paling banyak berada pada angka 1 dan 2.
# - Yang menarik disini adalah dimana ada rumah yang tidak ada kamar mandinya atau jumlahnya 0
# - Nilai outlier sendiri lumayan banyak.

# In[14]:


#Univariate analysis sqft_living
#Melihat distribusi dari sqft_living
f = plt.figure(figsize=(12,4))

f.add_subplot(1,2,1)
df['sqft_living'].plot(kind='kde')

f.add_subplot(1,2,2)
plt.boxplot(df['sqft_living'])
plt.show()


# - Density dari distribusi luas rumah berada di sekitar angka 2000an.
# - Banyak terdapat outliers.

# In[15]:


#Univariate analysis grade
#Melihat distribusi dari grade
f = plt.figure(figsize=(12,4))

f.add_subplot(1,2,1)
sns.countplot(df['grade'])

f.add_subplot(1,2,2)
plt.boxplot(df['grade'])
plt.show()


# - Sebagian besar rumah di County King US memiliki grade 7 dan 8.
# - Dilihat dari boxplot, data memiliki beberapa outliers.

# In[16]:


#Univariate analysis yr_built
#Melihat distribusi dari yr_built
f = plt.figure(figsize=(20,8))

f.add_subplot(1,2,1)
sns.countplot(df['yr_built'])

f.add_subplot(1,2,2)
plt.boxplot(df['yr_built'])
plt.show()


# - Dapat dilihat bahwa semakin tua umur dari rumah, maka semakin sedikit orang yang menjual rumahnya tersebut.
# - Density terdapat di sekitar tahun 1980an.
# - Data tidak memiliki outliers.

# In[17]:


#Bivariate analysis antara independent variable dan dependent variable
#Melihat hubungan antara independent dan dependent
#Menggunakan pairplot
plt.figure(figsize=(10,8))
sns.pairplot(data=df, x_vars=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'yr_built'], y_vars=['price'], size=5, aspect=0.75)
plt.show()


# In[19]:


# Mengetahui nilai korelasi dari independent variable dan dependent variable
df.corr().style.background_gradient().format("{:.2f}")


# - Dari tabel korelasi diatas, dapat dilihat bahwa sqft_living mempunyai hubungan linear positif yang sangat kuat dengan price jika dibandingkan yang lain.
# - Nilai korelasi yr_built hampir mendekati nol yang menandakan bahwa usia rumah tidak mempengaruhi pada harga rumah.

# # Modelling

# ## Model SVR

# ## 1. Import Libraries
# pandas: Untuk memuat dan memanipulasi data dalam format tabel.
# 
# numpy: Untuk manipulasi array.
# 
# train_test_split: Untuk membagi data menjadi set pelatihan (training) dan pengujian (testing).
# 
# StandardScaler: Untuk standarisasi data, membuat distribusi fitur memiliki mean 0 dan standar deviasi 1.
# 
# SVR: Algoritma Support Vector Regression dari sklearn.
# 
# mean_squared_error dan r2_score: Metrik evaluasi regresi.
# 
# matplotlib.pyplot: Untuk visualisasi hasil.

# In[31]:


#Recall data kita
df.head()


# ### Load library

# In[49]:


from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


# ## 2. Load Dataset
# pd.read_csv: Memuat dataset dari file CSV.
# 
# usecols: Hanya memuat kolom tertentu dari dataset. Kolom yang dipilih adalah:
# 
# bedrooms: Jumlah kamar tidur.
# 
# bathrooms: Jumlah kamar mandi.
# 
# sqft_living: Luas bangunan (dalam kaki persegi).
# 
# grade: Penilaian kualitas rumah.
# 
# price: Harga rumah (target prediksi).
# 
# yr_built: Tahun rumah dibangun.
# 

# In[34]:


# Load data
df = pd.read_csv("/content/kc_house_data.csv")


# ## 3. Bagi Data Menjadi Training dan Testing
# train_test_split: Membagi dataset menjadi dua bagian:
# 
# Training set (80%): Digunakan untuk melatih model.
# 
# Testing set (20%): Digunakan untuk menguji model.
# 
# random_state=42: Menjamin hasil pembagian data selalu sama setiap kali dijalankan.

# In[35]:


# Pertama, buat variabel x dan y
x = df.drop(columns='price')
y = df['price']


# In[36]:


# Kedua, kita split data menjadi training and testing dengan porsi 80:20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)


# In[37]:


# Pastikan hanya kolom numerik yang dipilih
x_train = x_train.select_dtypes(include=['number'])
x_test = x_test.select_dtypes(include=['number'])


# In[38]:


print(x_train.isnull().sum())  # Mengecek jumlah nilai yang hilang di setiap kolom
print(x_test.isnull().sum())   # Mengecek jumlah nilai yang hilang di setiap kolom


# In[39]:


x_train = x_train.fillna(x_train.mean())  # Isi dengan rata-rata
x_test = x_test.fillna(x_test.mean())    # Isi dengan rata-rata


# In[40]:


print(x_train.dtypes)  # Menampilkan tipe data untuk setiap kolom


# ## 4. Standarisasi Fitur dan Target
# StandardScaler: Skala fitur dan target agar nilai berada dalam distribusi normal dengan mean 0 dan standar deviasi 1.
# 
# fit_transform: Menyesuaikan skala dengan data training dan langsung mengubahnya.
# 
# transform: Menggunakan skala yang sudah dipelajari pada data lain (data testing).
# 
# reshape(-1, 1): Mengubah data target (y_train) menjadi bentuk 2D untuk StandardScaler.
# 
# .ravel(): Mengembalikan data ke bentuk 1D setelah transformasi.

# In[41]:


# Inisialisasi scaler
scaler = StandardScaler()

# Standarisasi data pelatihan dan pengujian
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[42]:


# Ketiga, buat objek SVR dengan kernel RBF
svr = SVR(kernel='rbf', C=100, epsilon=0.1)


# ## 5. Buat dan Latih Model SVR
# SVR:kernel='rbf': Kernel Radial Basis Function untuk menangani hubungan non-linear.
# 
# C=100: Parameter regularisasi; nilai lebih besar memperbolehkan model lebih fleksibel.
# 
# epsilon=0.1: Margin toleransi kesalahan; prediksi dianggap benar jika berada dalam
# ±0. dari nilai sebenarnya (setelah scaling).
# 
# fit: Melatih model menggunakan data training.

# In[43]:


# Keempat, latih model menggunakan data training yang telah distandarisasi
svr.fit(x_train_scaled, y_train)


# ## 6. Evaluasi Model
# mean_squared_error (MSE):
# 
# Rata-rata kesalahan kuadrat antara prediksi dan nilai aktual.
# 
# Nilai lebih kecil menunjukkan model yang lebih baik.
# 
# r2_score (R2):
# 
# Mengukur seberapa baik model menjelaskan variansi data target.
# 
# Nilai 𝑅2 berkisar antara 0 hingga 1, di mana 1 adalah yang terbaik.

# In[44]:


# Kelima, prediksi dengan data testing
y_pred = svr.predict(x_test_scaled)


# In[45]:


# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)


# In[46]:


# Contoh input untuk rumah Joko (19 fitur yang sesuai dengan data pelatihan)
joko_house = [[3, 2, 1800, 2, 7, 1970, 1000, 2000, 3000, 4, 0, 0, 1, 0, 0, 0, 1, 1, 0]]

# Standarisasi input rumah Joko
joko_house_scaled = scaler.transform(joko_house)

# Prediksi harga menggunakan model SVR
predicted_price = svr.predict(joko_house_scaled)

# Output hasil prediksi
print("Prediksi harga rumah Joko:", predicted_price[0])


# - Yeay! Harga rumah idaman Joko dan istirnya adalah sekitar 489442 US$

# In[50]:


import pickle
filename = '/content/kc_house_data.sav'
pickle.dump(SVR, open(filename, 'wb'))

