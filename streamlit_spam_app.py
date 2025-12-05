import streamlit as st
import pickle
import re
import string
import pandas as pd

# Load model dan vectorizer
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

# Mapping label dan balasan
label_dict = {
    "ham": "✅ Bukan Spam",
    "spam": "❌ SPAM"
}
response_dict = {
    "ham": "Email ini ditandai sebagai email legit.",
    "spam": "Email ini ditandai sebagai email spam."
}

# UI Streamlit
st.title("📧 Spam Detection App")
st.write("Masukkan teks email atau pesan dan model akan memprediksi apakah itu spam atau bukan.")

user_input = st.text_area("Masukkan teks di sini:")

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        cleaned = preprocess(user_input)
        vect_text = vectorizer.transform([cleaned])
        prediction = model.predict(vect_text)[0]

        # Tampilkan probabilitas jika model mendukung
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vect_text)[0]
            st.write(f"📊 **Probabilitas:** Bukan Spam = `{proba[0]:.4f}`, Spam = `{proba[1]:.4f}`")

        # Tampilkan hasil akhir
        st.markdown(f"### 📩 Hasil: {label_dict[prediction]}")
        st.info(response_dict[prediction])
