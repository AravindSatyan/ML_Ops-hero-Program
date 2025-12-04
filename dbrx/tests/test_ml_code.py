import joblib
import pandas as pd

def main_test():

    model = joblib.load("artifacts/telco_churn_model.joblib")

    sample = pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 80.5,
                "TotalCharges": 400.0,
            }
        ]
    )

    proba = model.predict_proba(sample)[:, 1][0]
    pred = model.predict(sample)[0]

    print(f"Churn probability: {proba:.3f}")
    print("Predicted label:", "Churn" if pred == 1 else "No churn")
# ensure probability is a native Python float and assert its type
    proba = float(proba)
    assert isinstance(proba, float), f"proba is not a float (got {type(proba)})"