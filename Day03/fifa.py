import pandas as pd
import numpy as np
import os

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Target Encoding
try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("Install category_encoders using:")
    print("pip install category_encoders")
    exit()

# Load Dataset

print("Loading Dataset...")

file_name = "fifa_world_cup_2026_player_performance.csv"

if not os.path.exists(file_name):
    print("Dataset not found!")
    exit()

df = pd.read_csv(file_name)

print("Dataset Loaded Successfully")
print(f"Rows : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("First 5 Records\n")
print(df.head())

print("Dataset Information\n")
print(df.info())

print("Statistical Summary\n")
print(df.describe())


# Missing Values
print("Checking Missing Values\n")

print(df.isnull().sum())

numeric_columns = [
    "age",
    "height_cm",
    "weight_kg",
    "minutes_played",
    "goals",
    "assists",
    "shots",
    "pass_accuracy",
    "stamina_score",
    "performance_score"
]

imputer = SimpleImputer(strategy="median")

df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

print("Missing Values Handled Successfully\n")

# Target Encoding

print("Applying Target Encoding\n")

encoder = TargetEncoder()

df["Team_Encoded"] = encoder.fit_transform(
    df["team"],
    df["tournament_rating"]
)

df["Position_Encoded"] = encoder.fit_transform(
    df["position"],
    df["tournament_rating"]
)

print("Target Encoding Completed")

# Log Transformation

print("Applying Log Transformation\n")

df["LogMinutes"] = np.log1p(df["minutes_played"])

print("Skewness :", df["LogMinutes"].skew())

# Feature Selection

print("Selecting Best Features\n")

features = [
    "age",
    "goals",
    "assists",
    "shots",
    "pass_accuracy",
    "stamina_score",
    "performance_score",
    "offensive_contribution",
    "defensive_contribution",
    "creativity_score",
    "consistency_score",
    "LogMinutes",
    "Team_Encoded",
    "Position_Encoded"
]

X = df[features]
y = df["tournament_rating"]

selector = SelectKBest(
    score_func=mutual_info_regression,
    k=6
)

selector.fit(X, y)

best_features = X.columns[
    selector.get_support()
].tolist()

print("\nBest Features")

for feature in best_features:
    print(feature)

# Train Test Split

print("Splitting Dataset\n")

X = df[best_features]
y = df["tournament_rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# Linear Regression

print("Training Model\n")

model = LinearRegression()

model.fit(X_train, y_train)

print("Model Trained Successfully")

# Prediction

prediction = model.predict(X_test)

# Create Result DataFrame
result = pd.DataFrame({
    "Player Name": df.loc[X_test.index, "player_name"].values,
    "Team": df.loc[X_test.index, "team"].values,
    "Position": df.loc[X_test.index, "position"].values,
    "Actual Rating": y_test.values,
    "Predicted Rating": prediction
})

# Sort by Predicted Rating
best_players = result.sort_values(
    by="Predicted Rating",
    ascending=False
)

print("\n========== TOP 10 BEST PLAYERS ==========\n")

print(best_players.head(10))


# Model Evaluation

print("Model Evaluation\n")

mae = mean_absolute_error(y_test, prediction)

mse = mean_squared_error(y_test, prediction)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, prediction)

print("Mean Absolute Error :", round(mae,2))
print("Mean Squared Error :", round(mse,2))
print("Root Mean Squared Error :", round(rmse,2))
print("R2 Score :", round(r2,2))

print("Program Completed Successfully\n")