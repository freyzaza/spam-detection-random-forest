import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.preprocessing import clean_text

DATA_PATH = "data/email_spam_indo.csv" 
TEXT_COL = "Pesan"
LABEL_COL = "Kategori"

VECTORIZER_OUT = "tfidf_vectorizer.pkl"
MODEL_OUT = "rf_model.pkl"

def main():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=str.strip)

    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"Kolom tidak ditemukan. Pastikan ada kolom '{TEXT_COL}' dan '{LABEL_COL}'.")

    df["clean_text"] = df[TEXT_COL].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df[LABEL_COL],
        test_size=0.20,
        stratify=df[LABEL_COL],
        random_state=42
    )

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )

    X_train_tf = tfidf.fit_transform(X_train)
    X_test_tf = tfidf.transform(X_test)

    X_train_rf = X_train_tf.toarray()
    X_test_rf = X_test_tf.toarray()

    rf = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=42
    )

    rf.fit(X_train_rf, y_train)
    y_pred = rf.predict(X_test_rf)

    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=rf.classes_, yticklabels=rf.classes_)
    plt.title("TF-IDF + Random Forest - Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(rf, f)

    with open(VECTORIZER_OUT, "wb") as f:
        pickle.dump(tfidf, f)

    print(f"\nSaved model -> {MODEL_OUT}")
    print(f"Saved vectorizer -> {VECTORIZER_OUT}")

if __name__ == "__main__":
    main()
