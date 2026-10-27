import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. FILE PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "emails.csv")
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 60)
print("SPAM EMAIL DETECTION - MODEL TRAINING")
print("=" * 60)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"emails.csv not found at:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Original columns:", df.columns.tolist())
print("Dataset shape:", df.shape)


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip().str.lower()

print("Cleaned columns:", df.columns.tolist())


# ============================================================
# 4. CHECK COLUMNS
# ============================================================

if "text" not in df.columns:
    raise ValueError("The dataset must contain a 'text' column.")

if "spam" not in df.columns:
    raise ValueError("The dataset must contain a 'spam' column.")


# ============================================================
# 5. CLEAN DATA
# ============================================================

df = df[["text", "spam"]].copy()

df = df.dropna(subset=["text", "spam"])

df["text"] = df["text"].astype(str)


print("\nData cleaned successfully!")
print("Rows after cleaning:", len(df))


# ============================================================
# 6. INPUT AND TARGET
# ============================================================

X = df["text"]
y = df["spam"]


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 8. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF conversion completed!")
print("Feature shape:", X_train_tfidf.shape)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

print("\nModel trained successfully!")


# ============================================================
# 10. EVALUATE MODEL
# ============================================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# ============================================================
# 11. SAVE MODEL
# ============================================================

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("\n" + "=" * 60)
print("FILES SAVED SUCCESSFULLY")
print("=" * 60)

print("\nModel:")
print(MODEL_PATH)

print("\nVectorizer:")
print(VECTORIZER_PATH)

print("\nPROGRAM COMPLETED!")