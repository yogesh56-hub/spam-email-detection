import os
import joblib
import streamlit as st

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

# CUSTOM CSS
st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

.spam-box {
    background-color: #ffe5e5;
    border: 2px solid #ff4b4b;
}

.safe-box {
    background-color: #e8f1ff;
    border: 2px solid #4a90e2;
}

.result-title {
    font-size: 30px;
    font-weight: bold;
}

.probability {
    font-size: 25px;
    font-weight: bold;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# FILE PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "spam_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "tfidf_vectorizer.pkl"
)

# LOAD MODEL
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):

    st.error("❌ Trained model not found.")

    st.info(
        "Please run this command first:\n\n"
        "`python train_model.py`"
    )

    st.stop()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# TITLE
st.markdown(
    '<div class="title">📧 Spam Email Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">NLP + Machine Learning</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter an email message below and the machine learning "
    "model will predict whether it is Spam or Not Spam."
)

# EMAIL INPUT
st.subheader("📩 Enter Email Message")

email = st.text_area(
    "",
    height=180,
    placeholder=(
        "Example: Congratulations! You have won a free prize! "
        "Click here to claim your reward..."
    )
)

# CHECK EMAIL BUTTON
check_button = st.button(
    "🔍 Check Email",
    use_container_width=True
)

# PREDICTION
if check_button:
    if not email.strip():
        st.warning("⚠️ Please enter an email message.")
    else:
        # Convert email to TF-IDF
        email_vector = vectorizer.transform([email])

        # Prediction
        prediction = model.predict(email_vector)[0]

        # Probability
        probabilities = model.predict_proba(email_vector)[0]

        # Find predicted class probability
        classes = model.classes_
        predicted_index = list(classes).index(prediction)
        probability = probabilities[predicted_index] * 100
        prediction_text = str(prediction).strip().lower()


        # SPAM RESULT
        if prediction_text in ["1", "spam", "true", "yes"]:
            st.markdown(
                """
                <div class="result-box spam-box">
                    <div class="result-title">
                        🚨 SPAM EMAIL
                    </div>
                    <p>This email is likely to be spam.</p>
                </div>
                """,
                unsafe_allow_html=True
            )


        # NOT SPAM RESULT
        else:
            st.markdown(
                """
                <div class="result-box safe-box">
                    <div class="result-title">
                        ✅ NOT SPAM
                    </div>
                    <p>This email appears to be legitimate.</p>
                </div>
                """,
                unsafe_allow_html=True
            )


        # PROBABILITY
        st.subheader("📊 Spam Probability")

        st.progress(
            int(probability)
        )

        st.markdown(
            f"""
            <div style="text-align:center;">
                <div class="probability">
                    {probability:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )




