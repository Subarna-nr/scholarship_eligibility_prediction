import argparse
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# SMOTE for oversampling the minority class
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    USE_SMOTE = True
except ImportError:
    print("⚠️ imbalanced-learn not installed. Run 'pip install imbalanced-learn' to enable SMOTE.")
    USE_SMOTE = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/train.csv')
    parser.add_argument('--out', type=str, default='models/best_model.joblib')
    parser.add_argument('--use_smote', action='store_true', help='Use SMOTE oversampling (requires imbalanced-learn)')
    args = parser.parse_args()

    # Load training data
    df = pd.read_csv(args.data)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # Identify numeric and categorical columns
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    # RandomForest with balanced class weights
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight='balanced'
    )

    # Build pipeline
    if args.use_smote and USE_SMOTE:
        # SMOTE oversampling + pipeline
        model = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('clf', clf)
        ])
    else:
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('clf', clf)
        ])

    # Fit model
    print("Training model...")
    model.fit(X, y)

    # Save model
    joblib.dump(model, args.out)
    print('✅ Saved model to', args.out)
