import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set page title
st.title("House Price Prediction using SVR")

# Load data
@st.cache_data
def load_data():
    # Pastikan file CSV ada di folder yang sama dengan script ini
    return pd.read_csv("kc_house_data.csv")

df = load_data()

# Sidebar for navigation
st.sidebar.header("Navigation")
options = st.sidebar.radio("Go to:", ["Data Exploration", "Model Training", "Prediction"])

# ==========================================
# Data Exploration
# ==========================================
if options == "Data Exploration":
    st.header("Exploratory Data Analysis")

    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Univariate Analysis")
    feature = st.selectbox("Select feature for analysis", df.columns)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[feature], kde=True, ax=ax[0])
    sns.boxplot(x=df[feature], ax=ax[1])
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    # Perbaikan: Hanya memproses kolom angka (numeric_only) agar tidak error
    sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ==========================================
# Model Training
# ==========================================
elif options == "Model Training":
    st.header("Train the SVR Model")

    # 1. PERBAIKAN: Filter HANYA kolom yang berisi angka (hindari kolom teks seperti 'date')
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # 2. Hapus 'price' dari daftar pilihan karena 'price' adalah hasil yang ingin diprediksi (target)
    if 'price' in numeric_columns:
        numeric_columns.remove('price')

    # Select features
    features = st.multiselect("Select features for prediction", 
                               options=numeric_columns, 
                               default=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'yr_built'])

    # --- TAMBAHKAN PENJAGA INI ---
    if len(features) == 0:
        st.warning("⚠️ Silakan pilih minimal satu fitur untuk melatih model!")
        st.stop()  # Menghentikan kode di bawahnya berjalan agar tidak error
    # -----------------------------

    x = df[features]
    y = df['price']
    x = df[features]
    y = df['price']

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

    # Standardize data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train SVR model
    svr = SVR(kernel='rbf', C=100, epsilon=0.1)
    svr.fit(x_train_scaled, y_train)

    # Evaluate model
    y_pred = svr.predict(x_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("**Model Performance:**")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

# ==========================================
# Prediction
# ==========================================
elif options == "Prediction":
    st.header("Predict House Price")

    # Gunakan HANYA fitur yang sama seperti saat melatih model
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'yr_built']

    st.write("Enter the features of the house:")
    input_features = []
    
    # Buat input form dan paksa mulai dari angka 0
    for col in features:
        if col == 'bathrooms':
            # Bathrooms mulai dari 0.0, format desimal
            val = st.number_input(f"{col}", value=0.0, step=0.5, format="%.2f")
        else:
            # Kolom lainnya mulai dari 0 murni, format integer tanpa koma
            val = st.number_input(f"{col}", value=0, step=1, format="%d")
            
        input_features.append(val)

    # Tambahkan tombol prediksi agar aplikasi tidak lag
    if st.button("Predict Price"):
        input_features = [input_features]
        
        # Ambil data X dan y
        X = df[features]
        y = df['price']
        
        # Latih scaler dengan data X yang benar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        input_scaled = scaler.transform(input_features)

        # Memunculkan animasi loading selagi model memproses
        with st.spinner("Training model and calculating prediction..."):
            svr = SVR(kernel='rbf', C=100, epsilon=0.1)
            svr.fit(X_scaled, y)
            predicted_price = svr.predict(input_scaled)

        # Perbaikan: Hilangkan koma pada hasil akhir dengan membulatkannya ke int() dan format ,.0f
        st.success(f"**Predicted Price:** ${int(predicted_price[0]):,.0f}")
