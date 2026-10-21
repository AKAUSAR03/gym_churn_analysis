import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix,
classification_report
)
from sklearn.base import clone
#1. Load dataset
df = pd.read_csv("gym_members_dataset.csv")
df.columns = df.columns.str.strip()
#2. Clean the target column
df["Churn"] = df["Churn"].astype(str).str.strip().str.lower()

target_mapping = {
"yes": 1,
"no": 0,
"1": 1,
"0": 0,
"true": 1,
"false": 0
}

df["Churn"] = df["Churn"].map(target_mapping)

#Remove rows where churn could not be interpreted

df = df.dropna(subset=["Churn"])
df["Churn"] = df["Churn"].astype(int)

#3. Remove identifiers and personal/date columns

columns_to_drop = [
"Member_ID",
"Name",
"Address",
"Phone_Number",
"Join_Date",
"Last_Visit_Date",
"Churn"
]

X = df.drop(
columns=[col for col in columns_to_drop if col in df.columns]
)
y = df["Churn"]

if X.empty:
   raise ValueError("No input features remain after removing unused columns.")

if y.nunique() < 2:
   raise ValueError(
     "The Churn column must contain both churn and non-churn examples."
)

#4. Identify numerical and categorical columns

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
exclude=["number"]
).columns.tolist()

#5. Create preprocessing steps

numeric_transformer = Pipeline(steps=[
("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
("imputer", SimpleImputer(strategy="most_frequent")),
("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
("num", numeric_transformer, numeric_features),
("cat", categorical_transformer, categorical_features)
])

#6. Split data into training and testing sets

class_counts = y.value_counts()
stratify_target = y if class_counts.min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42,
stratify=stratify_target
)

#7. Build the Random Forest model

random_forest = Pipeline(steps=[
("preprocessor", clone(preprocessor)),
("classifier", RandomForestClassifier(
n_estimators=200,
random_state=42,
class_weight="balanced"
))
])

#8. Build the Logistic Regression model for comparison

logistic_regression = Pipeline(steps=[
("preprocessor", clone(preprocessor)),
("classifier", LogisticRegression(
max_iter=1000,
class_weight="balanced",
random_state=42
))
])

#9. Train both models using the same training data

random_forest.fit(X_train, y_train)
logistic_regression.fit(X_train, y_train)

#10. Evaluate a model and collect its metrics

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    return {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, predictions, zero_division=0)
    ),
    "confusion_matrix": confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).tolist(),
    "classification_report": classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Retained (No Churn)", "Churned"],
        output_dict=True,
        zero_division=0
    )
}

rf_metrics = evaluate_model(random_forest, X_test, y_test)
lr_metrics = evaluate_model(logistic_regression, X_test, y_test)

#11. Save the trained Random Forest and evaluation results

model_data = {
"pipeline": random_forest,
"feature_columns": X.columns.tolist(),
"evaluation": rf_metrics,
"comparison": {
"Random Forest": rf_metrics,
"Logistic Regression": lr_metrics
},
"test_size": len(X_test)
}

joblib.dump(model_data, "churn_pipeline.pkl")

#12. Display evaluation results in the terminal

print("\nMODEL TRAINING COMPLETED")
print("-" * 40)

print("\nRandom Forest Evaluation")
print(f"Accuracy : {rf_metrics['accuracy']:.2%}")
print(f"Precision: {rf_metrics['precision']:.2%}")
print(f"Recall   : {rf_metrics['recall']:.2%}")
print(f"F1-score : {rf_metrics['f1_score']:.2%}")
print("\nConfusion Matrix:")
print(pd.DataFrame(
rf_metrics["confusion_matrix"],
index=["Actual Retained", "Actual Churned"],
columns=["Predicted Retained", "Predicted Churned"]
))

print("\nLogistic Regression Evaluation")
print(f"Accuracy : {lr_metrics['accuracy']:.2%}")
print(f"Precision: {lr_metrics['precision']:.2%}")
print(f"Recall   : {lr_metrics['recall']:.2%}")
print(f"F1-score : {lr_metrics['f1_score']:.2%}")

print("\nModel comparison:")
print(pd.DataFrame({
"Random Forest": rf_metrics,
"Logistic Regression": lr_metrics
}).loc[["accuracy", "precision", "recall", "f1_score"]])

print("\nSaved model and evaluation results to churn_pipeline.pkl")