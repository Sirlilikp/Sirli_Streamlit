import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Flight Loyalty Prediction",
    layout="wide"
)

st.title("✈️ Flight Loyalty Prediction App")
st.write("Prediksi **tingkat loyalitas pelanggan (FFP_TIER)** berbasis data penerbangan")

# ======================
# Upload Dataset
# ======================
uploaded_file = st.sidebar.file_uploader(
    "Upload dataset flight (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload file **flight.csv** terlebih dahulu")
    st.stop()

# ======================
# Load Data
# ======================
data = pd.read_csv(uploaded_file)

# ======================
# Data Cleaning
# ======================
DROP_COLS = [
    "MEMBER_NO",
    "FFP_DATE",
    "FIRST_FLIGHT_DATE",
    "LAST_FLIGHT_DATE",
    "LOAD_TIME"
]

existing_cols = [c for c in DROP_COLS if c in data.columns]
data = data.drop(columns=existing_cols)

data["WORK_CITY"] = data["WORK_CITY"].replace(".", "Unknown")
data["AGE"] = data["AGE"].fillna(data["AGE"].median())

target_column = "FFP_TIER"

# ======================
# Sidebar Train
# ======================
if st.sidebar.button("Train Model"):
    with st.spinner("Training loyalty prediction model..."):

        X = data.drop(columns=[target_column])
        y = data[target_column]

        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42
        )

        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))

        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["features"] = X.columns
        st.session_state["accuracy"] = acc

    st.success("Model berhasil dilatih 🎉")

# ======================
# Dashboard
# ======================
if "model" in st.session_state:
    model = st.session_state["model"]
    scaler = st.session_state["scaler"]
    features = st.session_state["features"]
    acc = st.session_state["accuracy"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Akurasi Model", f"{acc:.2f}")
    col2.metric("Jumlah Member", data.shape[0])
    col3.metric("Jumlah Fitur", len(features))

    # ======================
    # Target Distribution
    # ======================
    st.subheader("📊 Distribusi Tingkat Loyalitas (FFP_TIER)")
    fig_tier = px.histogram(
        data,
        x="FFP_TIER",
        color="FFP_TIER"
    )
    st.plotly_chart(fig_tier, use_container_width=True)

    # ======================
    # Feature Importance
    # ======================
    st.subheader("⭐ Faktor Penentu Loyalitas")
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(15)

    fig_imp = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # ======================
    # Prediction
    # ======================
    st.subheader("🔮 Prediksi Loyalitas Member Baru")

    input_data = {}
    cols = st.columns(3)
    feature_source = data.drop(columns=[target_column])

    for i, col in enumerate(feature_source.columns):
        if feature_source[col].dtype == "object":
            input_data[col] = cols[i % 3].selectbox(
                col,
                feature_source[col].unique()
            )
        else:
            input_data[col] = cols[i % 3].number_input(
                col,
                value=float(feature_source[col].median())
            )

    if st.button("Prediksi Loyalitas"):
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=features, fill_value=0)
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        st.success(f"🎯 **Prediksi FFP Tier: {prediction}**")

import streamlit as st

def app():
    st.title("✈️ Flight Loyalty Prediction")
    st.write("Aplikasi prediksi loyalitas pelanggan")
    # seluruh kode prediksi kamu DI SINI
