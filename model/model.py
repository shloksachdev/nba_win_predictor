# Importing libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Remove existing model if it exists
try:
    os.remove("trainedmodel1.pkl")
except FileNotFoundError:
    pass

# Load and prepare data
print("Loading Data...")
df = pd.read_csv("../data/csv/dataset.csv")
print("Data Loaded Successfully!")

# Drop unnecessary columns
df = df.drop(["date", "home_team", "away_team"], axis=1)

# Drop specific stats (if any are listed)
stats_to_drop = []
for stat in stats_to_drop:
    df = df.drop([f"home_{stat}", f"away_{stat}"], axis=1)

# Define features (X) and target (y)
X = df.drop("winning_team", axis=1)
y = df["winning_team"]

# Split the data
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost model
print("Training model... please wait.")
xgb = XGBClassifier(
    colsample_bytree=0.25648728122860365,
    eval_metric="logloss",
    gamma=9.672827578666489,
    learning_rate=0.03670169009052889,
    max_depth=14,
    min_child_weight=4,
    n_estimators=938,
    objective="binary:logistic",
    reg_alpha=0.39672445352573993,
    reg_lambda=3.0454849867410543,
    subsample=0.6706647946130068,
    random_state=42,
)
xgb.fit(X_train, y_train)
print("Model training complete!")

# Evaluate model
print("Evaluating the model...")
y_pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(xgb, "trainedmodel1.pkl")
print("\nModel saved successfully as 'trainedmodel1.pkl' ✅")

# Analyze and print complete feature importance
print("\nFeature Importances:")

# Make sure Pandas does not truncate output
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)

feature_importances = xgb.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": feature_importances
})

# Sort by Importance
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Print full importance
print(importance_df)

# (Optional) Reset Pandas options to default after printing
pd.reset_option("display.max_columns")
pd.reset_option("display.width")
pd.reset_option("display.max_rows")
