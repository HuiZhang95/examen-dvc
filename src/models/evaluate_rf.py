import pandas as pd 
import numpy as np
from joblib import load
import json
from pathlib import Path

from sklearn.metrics import accuracy_score
import pickle

X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

def main(repo_path):
    model_filename = repo_path / "models/train_rf.pkl"
    model = pickle.load(open(model_filename, 'rb'))

    accuracy = model.score(X_test, y_test)
    metrics = {"accuracy": accuracy}
    accuracy_path = repo_path / "metrics/accuracy.json"
    accuracy_path.write_text(json.dumps(metrics))

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)