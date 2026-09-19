import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------- MLflow Connection ----------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")


# ---------------- Prepare the Same TF-IDF Vectorizer ----------------

df = pd.read_csv("Sentiment.csv")

df = df.dropna(subset=["body"])
df = df.drop_duplicates()

vectorizer = TfidfVectorizer(max_features=5000)

vectorizer.fit_transform(df["body"])


# ---------------- Unseen Data ----------------

unseen_text = [
    "I really enjoyed this product, it is excellent!",
    "This is a terrible experience and I am very disappointed.",
    "The product is okay, nothing special."
]

unseen_features = vectorizer.transform(unseen_text)


# ---------------- Load Production Model ----------------

print("\n========== Production Model ==========")

production_model = mlflow.sklearn.load_model(
    "models:/Sentiment-Best-Model@production"
)

production_predictions = production_model.predict(unseen_features)

for text, prediction in zip(unseen_text, production_predictions):
    print("\nText:", text)
    print("Production Prediction:", prediction)


# ---------------- Load Previous Model Versions ----------------

print("\n========== Model Version Comparison ==========")

version_1_model = mlflow.sklearn.load_model(
    "models:/Sentiment-Best-Model/1"
)

version_2_model = mlflow.sklearn.load_model(
    "models:/Sentiment-Best-Model/2"
)

version_1_predictions = version_1_model.predict(unseen_features)
version_2_predictions = version_2_model.predict(unseen_features)


# ---------------- Compare Predictions ----------------

for i, text in enumerate(unseen_text):

    print("\nText:", text)
    print("Version 1 Prediction:", version_1_predictions[i])
    print("Version 2 Prediction:", version_2_predictions[i])
    print("Production Prediction:", production_predictions[i])


print("\n========== Production Inference Completed ==========")