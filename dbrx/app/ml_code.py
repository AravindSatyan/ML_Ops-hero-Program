import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

def main():
    # ---------- 1. Load data ----------
    DATA_PATH = "dbrx/telco-customer-churn.csv"
    df = pd.read_csv(DATA_PATH)

    # ---------- 2. Basic cleaning ----------

    # Convert TotalCharges to numeric (it comes as string with some blanks)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Drop rows where TotalCharges can't be converted (usually a tiny number)
        df = df.dropna(subset=["TotalCharges"])

    # Drop customerID – it's just an identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Target column
    TARGET_COL = "Churn"
    y = df[TARGET_COL].map({"Yes": 1, "No": 0})  # binary 0/1
    X = df.drop(columns=[TARGET_COL]) #we are removing it from the dataset, bc this is the target variable, it should not be part of training

    # ---------- 3. Feature types ----------

    # Infer numeric and categorical columns
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # ---------- 4. Preprocessing + model pipeline ----------

    numeric_transformer = StandardScaler() #why? not 34.0f
    categorical_transformer = OneHotEncoder(handle_unknown="ignore") #transform categorical to numeric

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = LogisticRegression(
        max_iter=1000,
        class_weight=None,  # Remove balancing to improve accuracy
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    # ---------- 5. Train / test split ----------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ---------- 6. Hyperparameter tuning ----------
    param_grid = {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__solver": ["liblinear", "lbfgs"],
    }
    grid = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    print("Training model with hyperparameter tuning...")
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # ---------- 7. Evaluate ----------
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"ROC-AUC:  {roc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["No churn", "Churn"]))

    print("Confusion matrix:")
    print(cm)

    # ---------- 8. Save model ----------
    os.makedirs("artifacts", exist_ok=True)
    MODEL_PATH = os.path.join("artifacts", "telco_churn_model.joblib")
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    
if __name__ == "__main__":
    main()