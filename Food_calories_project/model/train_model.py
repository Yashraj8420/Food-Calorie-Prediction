import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
import os

# Load dataset
df = pd.read_csv("../data/food_data.csv")

# Clean food names
df["food"] = df["food"].astype(str).str.strip().str.lower()

# Features and target
X = df[['food', 'quantity']]
y = df['calories']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('food', OneHotEncoder(handle_unknown='ignore'), ['food'])
    ],
    remainder='passthrough'
)

# Model Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Evaluate model
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print("Model trained successfully!")
print("Mean Absolute Error (MAE):", round(mae, 3))

# Save model in model folder
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully")