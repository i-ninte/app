import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
data = pd.read_csv('water_potability.csv')

# Select features and target
X = data[['ph', 'turbidity']]
y = data['class']

# Handle missing values if necessary
X['ph'].fillna(X['ph'].mean(), inplace=True)
X['turbidity'].fillna(X['turbidity'].mean(), inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the GBM model
gbm = GradientBoostingClassifier()
gbm.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = gbm.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model using joblib
joblib_file = "gbm_model.pkl"
joblib.dump(gbm, joblib_file)
print(f"Model saved to {joblib_file}")

# Function to take user input and make a prediction
def predict_water_quality(ph, turbidity):
    # Load the saved model
    model = joblib.load(joblib_file)

    # Create a DataFrame from user input
    user_input = pd.DataFrame({'ph': [ph], 'turbidity': [turbidity]})

    # Predict the class
    prediction = model.predict(user_input)
    return "Potable" if prediction[0] == 1 else "Not Potable"

if __name__ == "__main__":
    # Example of taking user input
    ph = float(input("Enter pH value: "))
    turbidity = float(input("Enter Turbidity value: "))

    # Predict water quality
    result = predict_water_quality(ph, turbidity)
    print(f"The predicted water quality is: {result}")
