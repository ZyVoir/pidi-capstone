import sys
import json
import argparse
import pandas as pd
import joblib

MODEL_PATH = "model/attrition_model.pkl"

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        sys.exit(1)

def predict_employee(input_data):
    model = load_model()
    if isinstance(input_data, dict):
        df_in = pd.DataFrame([input_data])
    elif isinstance(input_data, str) and input_data.endswith('.csv'):
        df_in = pd.read_csv(input_data)
    else:
        raise ValueError("Input data must be a dictionary or path to a CSV file.")
    
    predictions = model.predict(df_in)
    probabilities = model.predict_proba(df_in)[:, 1]
    
    results = []
    for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
        status = "HIGH RISK (Attrition Likely)" if pred == 1 else "LOW RISK (Retention Likely)"
        res = {
            "EmployeeIndex": i,
            "Prediction": int(pred),
            "Status": status,
            "AttritionProbability": f"{proba * 100:.2f}%"
        }
        results.append(res)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Employee Attrition Risk for PT Jaya Jaya Maju")
    parser.add_argument("--json", type=str, help="JSON string containing employee feature data")
    parser.add_argument("--file", type=str, help="Path to CSV file containing employee feature data")
    
    args = parser.parse_args()
    
    if args.json:
        data = json.loads(args.json)
        res = predict_employee(data)
        print(json.dumps(res, indent=2))
    elif args.file:
        res = predict_employee(args.file)
        print(json.dumps(res, indent=2))
    else:
        # Sample employee demonstration
        sample_employee = {
            "Age": 29,
            "BusinessTravel": "Travel_Frequently",
            "DailyRate": 450,
            "Department": "Sales",
            "DistanceFromHome": 22,
            "Education": 3,
            "EducationField": "Marketing",
            "EnvironmentSatisfaction": 1,
            "Gender": "Male",
            "HourlyRate": 55,
            "JobInvolvement": 2,
            "JobLevel": 1,
            "JobRole": "Sales Executive",
            "JobSatisfaction": 1,
            "MaritalStatus": "Single",
            "MonthlyIncome": 2500,
            "MonthlyRate": 12000,
            "NumCompaniesWorked": 4,
            "OverTime": "Yes",
            "PercentSalaryHike": 11,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 2,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 4,
            "TrainingTimesLastYear": 2,
            "WorkLifeBalance": 1,
            "YearsAtCompany": 2,
            "YearsInCurrentRole": 1,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 1
        }
        print("--- Running Sample Prediction ---")
        res = predict_employee(sample_employee)
        print(json.dumps(res, indent=2))
