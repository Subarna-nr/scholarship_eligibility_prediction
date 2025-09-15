import argparse
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/best_model.joblib')
    parser.add_argument('--data', type=str, default='data/processed/test.csv')
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.data)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    preds = model.predict(X)
    print("✅ Classification Report:")
    print(classification_report(y, preds))
    print('Confusion matrix:\n', confusion_matrix(y, preds))
