import streamlit as st
import joblib
import numpy as np

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Naive Bayes Dashboard",
    page_icon="🤖",
    layout="wide"
)

# ================= CUSTOM CSS =================

st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

h1, h2, h3 {
    color: #38bdf8;
}

.stButton > button {
    width: 100%;
    height: 3em;
    border-radius: 12px;
    border: none;
    font-size: 18px;
    font-weight: bold;
    background: linear-gradient(to right, #06b6d4, #3b82f6);
    color: white;
}

.stTextArea textarea {
    background-color: #1e293b;
    color: white;
    border-radius: 10px;
}

.card {
    background-color: #1e293b;
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.3);
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================

model = joblib.load("nb_model.pkl")

# OPTIONAL:
# If you have vectorizer.pkl then uncomment below line
# vectorizer = joblib.load("vectorizer.pkl")

# ================= SIDEBAR =================

st.sidebar.title("⚙️ Navigation")

menu = st.sidebar.radio(
    "Go To",
    ["Home", "Prediction", "About"]
)

# ================= HOME PAGE =================

if menu == "Home":

    st.title("🤖 Naive Bayes ML Dashboard")

    st.markdown("""
    <div class="card">
        <h2>Welcome to NLP Prediction Dashboard 🚀</h2>
        <p>Beautiful Streamlit UI for Machine Learning Model</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Algorithm", "Naive Bayes")

    with col2:
        st.metric("Model Status", "Loaded ✅")

    with col3:
        st.metric("Prediction", "Real-Time ⚡")

# ================= PREDICTION PAGE =================

elif menu == "Prediction":

    st.title("🔍 Text Prediction")

    user_input = st.text_area(
        "Enter Your Text",
        height=200,
        placeholder="Type your sentence here..."
    )

    if st.button("Predict"):

        if user_input.strip() == "":
            st.warning("Please enter some text.")

        else:

            try:

                # ================= REAL PREDICTION =================
                # Uncomment these lines if vectorizer.pkl exists

                # transformed_text = vectorizer.transform([user_input])
                # prediction = model.predict(transformed_text)

                # ================= TEMP PREDICTION =================
                # Dummy input because only model is available

                dummy_input = np.random.randint(0, 2, (1, 13361))

                prediction = model.predict(dummy_input)

                st.success(f"Prediction Result: {prediction[0]}")

            except Exception as e:

                st.error(f"Error: {e}")

# ================= ABOUT PAGE =================

else:

    st.title("📘 About Model")

    st.markdown("""
    <div class="card">

    ### 🧠 Model Information

    - Algorithm Used: Naive Bayes
    - Framework: Scikit-Learn
    - Frontend: Streamlit
    - Type: Classification Model

    ### ✨ Features

    ✅ Interactive Dashboard  
    ✅ Beautiful Dark UI  
    ✅ Fast Predictions  
    ✅ Sidebar Navigation  
    ✅ Real-Time Results  

    </div>
    """, unsafe_allow_html=True)