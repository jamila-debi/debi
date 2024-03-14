import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import sys
from train import LR, x_val, y_val

y_val_pred = LR.predict(x_val)

# Calculate validation accuracy
val_acc = accuracy_score(y_val, y_val_pred)
sentence = "Validation Accuracy: "+ str(val_acc)
with open("validation_accuracy.txt", 'w') as file:
    file.write(sentence)