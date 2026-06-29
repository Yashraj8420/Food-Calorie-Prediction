import streamlit as st
import pandas as pd
import pickle
import os

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "food_data.csv")

# ---------- Check Files ----------
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found!")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("CSV file not found!")
    st.stop()

# ---------- Load Model & Data ----------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

df = pd.read_csv(DATA_PATH)
df["food"] = df["food"].astype(str).str.strip().str.lower()

# ---------- Styling ----------
st.markdown("""
<style>

.stApp{
    background-image:url("https://images.unsplash.com/photo-1504674900247-0877df9cc836");
    background-size:cover;
    background-position:center;
    background-repeat:no-repeat;
    filter:saturate(180%);
}

.block-container{
    background:rgba(255,255,255,0.90);
    padding:30px;
    border-radius:20px;
}

/* Make all text black */
h1,h2,h3,h4,h5,h6,label,p,span{
    color:black !important;
}

/* Predict Button */
.stButton>button{
    background-color:#ff6b35;
    color:white;
    font-weight:bold;
    border:none;
    border-radius:10px;
    width:100%;
    height:45px;
}

.stButton>button:hover{
    background-color:#e65a25;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown("""
<h1 style="
text-align:center;
color:black;
font-size:50px;
font-weight:bold;
">
🍽️ Food Calorie Predictor
</h1>
""", unsafe_allow_html=True)

# ---------- Input ----------
food = st.selectbox(
    "Select Food",
    sorted(df["food"].unique())
)

quantity = st.number_input(
    "Quantity (1 = 100g)",
    min_value=1,
    value=1
)

# ---------- Prediction ----------
if st.button("🔍 Predict"):

    actual_quantity = quantity * 100

    input_df = pd.DataFrame({
        "food": [food],
        "quantity": [actual_quantity]
    })

    calories = model.predict(input_df)[0]

    food_info = df[df["food"] == food].iloc[0]

    protein = food_info["protein"] * actual_quantity / 100
    carbs = food_info["carbs"] * actual_quantity / 100
    fat = food_info["fat"] * actual_quantity / 100

    st.success(f"🔥 Calories : {calories:.2f} kcal")

    c1, c2, c3 = st.columns(3)

    c1.metric("💪 Protein", f"{protein:.2f} g")
    c2.metric("🍞 Carbs", f"{carbs:.2f} g")
    c3.metric("🧈 Fat", f"{fat:.2f} g")

    if calories > 300:
        st.warning("⚠️ High calorie meal! Consider lighter options.")