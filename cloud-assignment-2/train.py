import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

# Load train data
#if len(sys.argv) == 2:
#    train_data = pd.read_csv(sys.argv[1])
#else:
train_data = pd.read_csv("train.csv")

# Drop rows with missing values
train_data.dropna(inplace=True)

# Drop unnecessary columns
train_data.drop(columns=['PassengerId', 'Name', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP'], inplace=True)

# Split data into features (x) and target (y)
x = train_data.drop("Transported", axis=1).astype(int)
y = train_data["Transported"].astype(int)

# Split data into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and train Logistic Regression model
LR = LogisticRegression()
LR.fit(x_train, y_train)

# Predict on validation set
y_train_pred = LR.predict(x_train)

# Calculate validation accuracy
train_acc = accuracy_score(y_train, y_train_pred)
sentence = "Training Accuracy: "+ str(train_acc)
with open("training_accuracy.txt", 'w') as file:
    file.write(sentence)


# Save the trained model (optional)
# You can use joblib or pickle to save the model for future use
# Example: joblib.dump(LR, "logistic_regression_model.joblib")
