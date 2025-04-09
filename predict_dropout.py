import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load("dropout_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function to make predictions
def predict_dropout(input_data):
    df = pd.DataFrame([input_data])
    df = scaler.transform(df)  # Apply scaling
    prediction = model.predict(df)
    return "Dropped Out" if prediction[0] == 1 else "Stayed"

# Example Usage
if __name__ == "__main__":
    student = {
        "Age": 18,
        "Gender": 1,  # 0 = Female, 1 = Male
        "Parental_Education": 2,  # Encoded value
        "Attendance": 75.0,
        "GPA": 3.0,
        "Extracurricular_Activities": 1,
        "Disciplinary_Actions": 0,
        "Socioeconomic_Status": 1
    }
    print("Prediction:", predict_dropout(student))
