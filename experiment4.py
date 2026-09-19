from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the Sentiment Analysis dataset
df = pd.read_csv("Sentiment.csv")

# Display dataset information
print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())


# Remove rows where body is missing
df = df.dropna(subset=["body"])

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset after cleaning:")
print("Shape:", df.shape)

print("\nMissing Values after cleaning:")
print(df.isnull().sum())
# Select required columns for sentiment analysis
df = df[["body", "score"]]

print("\nSelected columns:")
print(df.head())

print("\nFinal Dataset Shape:", df.shape)
# Convert score into sentiment labels
def get_sentiment(score):
    if score >= 30:
        return "positive"
    elif score <= 10:
        return "negative"
    else:
        return "neutral"

df["sentiment"] = df["score"].apply(get_sentiment)

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

print("\nDataset with Sentiment:")
print(df[["body", "score", "sentiment"]].head())
# Convert text into numerical features using TF-IDF
# Convert text into numerical features using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["body"])
y = df["sentiment"]

print("\nTF-IDF Shape:", X.shape)
print("Target Shape:", y.shape)
# Split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
# Train Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression(max_iter=1000, class_weight="balanced")

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# ---------------- MLflow Tracking ----------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Sentiment-Analysis-Experiment")

with mlflow.start_run():

    # Log parameters
    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("class_weight", "balanced")

    # Log accuracy
    mlflow.log_metric("accuracy", accuracy)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["negative", "neutral", "positive"],
        yticklabels=["negative", "neutral", "positive"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")
    plt.close()

    # Log confusion matrix as artifact
    mlflow.log_artifact("confusion_matrix.png")

    # Log trained model
    mlflow.sklearn.log_model(model, "sentiment_model")

    print("\nMLflow run completed successfully!")
    # ---------------- Decision Tree Model ----------------

dt_model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Accuracy:", dt_accuracy)

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_pred))

# MLflow tracking for Decision Tree
mlflow.set_experiment("Sentiment-Analysis-Experiment")

with mlflow.start_run():

    mlflow.log_param("model", "Decision Tree")
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("class_weight", "balanced")

    mlflow.log_metric("accuracy", dt_accuracy)

    # Confusion Matrix
    dt_cm = confusion_matrix(y_test, dt_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        dt_cm,
        annot=True,
        fmt="d",
        xticklabels=["negative", "neutral", "positive"],
        yticklabels=["negative", "neutral", "positive"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Decision Tree Confusion Matrix")

    plt.savefig("decision_tree_confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("decision_tree_confusion_matrix.png")

    mlflow.sklearn.log_model(
    dt_model,
    name="decision_tree_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
    print("\nDecision Tree MLflow run completed successfully!")
    # ---------------- Random Forest Model ----------------

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# MLflow tracking for Random Forest
mlflow.set_experiment("Sentiment-Analysis-Experiment")

with mlflow.start_run():

    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("class_weight", "balanced")

    mlflow.log_metric("accuracy", rf_accuracy)

    # Confusion Matrix
    rf_cm = confusion_matrix(y_test, rf_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        rf_cm,
        annot=True,
        fmt="d",
        xticklabels=["negative", "neutral", "positive"],
        yticklabels=["negative", "neutral", "positive"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Random Forest Confusion Matrix")

    plt.savefig("random_forest_confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("random_forest_confusion_matrix.png")

    # Log trained model
    mlflow.sklearn.log_model(
        rf_model,
        name="random_forest_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print("\nRandom Forest MLflow run completed successfully!")
    # ---------------- Hyperparameter Tuning ----------------

print("\n========== Hyperparameter Tuning ==========")

# Different C values to test
c_values = [0.1, 1, 10]

for c in c_values:

    tuned_model = LogisticRegression(
        C=c,
        max_iter=1000,
        class_weight="balanced"
    )

    tuned_model.fit(X_train, y_train)

    tuned_pred = tuned_model.predict(X_test)

    tuned_accuracy = accuracy_score(y_test, tuned_pred)

    print(f"\nLogistic Regression with C={c}")
    print("Accuracy:", tuned_accuracy)

    # Log tuning experiment in MLflow
    mlflow.set_experiment("Sentiment-Analysis-Experiment")

    with mlflow.start_run(run_name=f"Logistic Regression C={c}"):

        mlflow.log_param("model", "Logistic Regression")
        mlflow.log_param("C", c)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")

        mlflow.log_metric("accuracy", tuned_accuracy)

        print(f"MLflow tuning run completed for C={c}")

print("\n========== Hyperparameter Tuning Completed ==========")
# ---------------- MLflow Model Registry ----------------

print("\n========== Model Registry ==========")

registered_model_name = "Sentiment-Best-Model"

# Register Version 1 - Logistic Regression C=1
registry_model_v1 = LogisticRegression(
    C=1,
    max_iter=1000,
    class_weight="balanced"
)

registry_model_v1.fit(X_train, y_train)

v1_pred = registry_model_v1.predict(X_test)
v1_accuracy = accuracy_score(y_test, v1_pred)

with mlflow.start_run(run_name="Registry-Version-1-C1"):

    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_param("C", 1)
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", v1_accuracy)

    mlflow.sklearn.log_model(
        registry_model_v1,
        name="sentiment_model",
        registered_model_name=registered_model_name
    )

    print("Registered Model Version 1 created")
    print("C=1 Accuracy:", v1_accuracy)


# Register Version 2 - Logistic Regression C=10
registry_model_v2 = LogisticRegression(
    C=10,
    max_iter=1000,
    class_weight="balanced"
)

registry_model_v2.fit(X_train, y_train)

v2_pred = registry_model_v2.predict(X_test)
v2_accuracy = accuracy_score(y_test, v2_pred)

with mlflow.start_run(run_name="Registry-Version-2-C10"):

    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_param("C", 10)
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", v2_accuracy)

    mlflow.sklearn.log_model(
        registry_model_v2,
        name="sentiment_model",
        registered_model_name=registered_model_name
    )

    print("Registered Model Version 2 created")
    print("C=10 Accuracy:", v2_accuracy)

print("\n========== Model Registry Completed ==========")
# ---------------- Production Model Inference ----------------

print("\n========== Production Model Inference ==========")

# Set MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the model assigned to the Production alias
production_model = mlflow.sklearn.load_model(
    "models:/Sentiment-Best-Model@Production"
)

# Unseen text data
unseen_text = [
    "I really enjoyed this product, it is excellent!",
    "This is a terrible experience and I am very disappointed.",
    "The product is okay, nothing special."
]

# Convert unseen text using the same TF-IDF vectorizer
unseen_features = vectorizer.transform(unseen_text)

# Production model predictions
production_predictions = production_model.predict(unseen_features)

print("\nProduction Model Predictions:")

for text, prediction in zip(unseen_text, production_predictions):
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {prediction}")
    print()

# Compare predictions from Version 1 and Version 2
version_1_model = mlflow.sklearn.load_model(
    "models:/Sentiment-Best-Model/1"
)

version_2_model = mlflow.sklearn.load_model(
    "models:/Sentiment-Best-Model/2"
)

version_1_predictions = version_1_model.predict(unseen_features)
version_2_predictions = version_2_model.predict(unseen_features)

print("========== Version Comparison ==========")

for i, text in enumerate(unseen_text):
    print(f"Text: {text}")
    print(f"Version 1 Prediction: {version_1_predictions[i]}")
    print(f"Version 2 Prediction: {version_2_predictions[i]}")
    print(f"Production Prediction: {production_predictions[i]}")
    print()

print("========== Production Inference Completed ==========")