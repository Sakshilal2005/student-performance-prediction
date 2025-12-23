import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv('data/student.csv')

# Define features (input) and target (output)
X = data[['hours_studied', 'attendance', 'previous_score']]  # multiple input features
y = data['final_score']  # target column

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a file
pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved successfully.")

