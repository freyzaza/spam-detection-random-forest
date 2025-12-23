import pickle
import streamlit as st

from src.preprocessing import clean_text

MODEL_PATH = "rf_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

st.set_page_config(page_title="Spam Detection", layout="centered")
st.title("Indonesian Email Spam Detection (TF-IDF + Random Forest)")

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()

# =========================================================
# Threshold Mode: Light / Moderate / Strict 
# =========================================================
threshold_map = {
    "Light": 0.35,     
    "Moderate": 0.60,  
    "Strict": 0.80     
}

mode = st.radio(
    "Mode deteksi spam",
    options=["Light", "Moderate", "Strict"],
    index=1,           
    horizontal=True
)

threshold = threshold_map[mode]

st.caption(
    f"Mode {mode} menggunakan threshold {threshold:.2f}. "
    "Email dianggap SPAM jika probabilitas spam ≥ threshold."
)

# =========================================================
# Input
# =========================================================
text = st.text_area("Masukkan teks email:", height=180)

def get_spam_probability(model, X_dense):
    """
    Ambil probabilitas spam secara aman berdasarkan model.classes_.
    Return (spam_proba or None, spam_label_name or None)
    """
    if not hasattr(model, "predict_proba"):
        return None, None

    proba = model.predict_proba(X_dense)[0]
    classes = list(model.classes_)


    spam_idx = None
    for i, c in enumerate(classes):
        c_norm = str(c).strip().lower()
        if c_norm in ["spam", "1", "true", "yes"]:
            spam_idx = i
            break

    if spam_idx is None and len(classes) == 2:
        spam_idx = 1

    if spam_idx is None:
        return None, None

    return float(proba[spam_idx]), classes[spam_idx]

if st.button("Predict"):
    cleaned = clean_text(text)

    X = vectorizer.transform([cleaned])
    X_dense = X.toarray()

    spam_proba, spam_label = get_spam_probability(model, X_dense)

    if spam_proba is not None:
        st.write(f"Probabilitas spam: {spam_proba:.4f}")
        is_spam = spam_proba >= threshold
    else:
        pred_label = model.predict(X_dense)[0]
        pred_norm = str(pred_label).strip().lower()
        is_spam = pred_norm in ["spam", "1", "true", "yes"]

    if is_spam:
        st.error("Email terdeteksi sebagai SPAM")
    else:
        st.success("Email terdeteksi sebagai Bukan Spam")
        