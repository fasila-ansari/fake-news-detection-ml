# ============================================================
# Fake News Detection using Machine Learning
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("===================================")
print("Fake News Detection Project Started")
print("===================================")

# ============================================================
# Create images folder
# ============================================================

os.makedirs("images", exist_ok=True)

# ============================================================
# Load datasets
# ============================================================

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

print("\nDatasets Loaded Successfully")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
df = pd.concat([fake, true])

print("\nDataset shape:", df.shape)

# ============================================================
# Use only necessary columns
# ============================================================

df = df[["title", "text", "label"]]

# Combine title and text
df["content"] = df["title"] + " " + df["text"]

# ============================================================
# Split features and target
# ============================================================

X = df["content"]
y = df["label"]

# ============================================================
# Train test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# Convert text to numerical features
# ============================================================

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# ============================================================
# Train models
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB()
}

results = {}

print("\nTraining models...")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print(f"{name} Accuracy:", accuracy)

# ============================================================
# Plot accuracy comparison
# ============================================================

plt.figure(figsize=(8,5))

sns.barplot(
    x=list(results.keys()),
    y=list(results.values())
)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.savefig("images/model_accuracy.png")

plt.show()

print("\nFake News Detection Project Completed!")