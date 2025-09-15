import argparse
import pandas as pd
import re
from sklearn.model_selection import train_test_split

def convert_income(val):
    """Convert income ranges like 'Upto 1.5L' to numeric rupees."""
    if pd.isna(val):
        return None
    val = str(val)
    nums = re.findall(r'\d+\.?\d*', val)
    if not nums:
        return None
    num = float(nums[0])
    if 'l' in val.lower():  # Lakhs to rupees
        num = num * 100000
    return num

def convert_percentage(val):
    """Convert '90-100' or '80-89' to midpoint."""
    if pd.isna(val):
        return None
    val = str(val)
    nums = re.findall(r'\d+\.?\d*', val)
    if len(nums) == 2:
        return (float(nums[0]) + float(nums[1])) / 2
    elif len(nums) == 1:
        return float(nums[0])
    return None

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop Name if present
    if 'Name' in df.columns:
        df.drop('Name', axis=1, inplace=True)

    # Rename Education Qualification to Education if needed
    if 'Education Qualification' in df.columns:
        df.rename(columns={'Education Qualification': 'Education'}, inplace=True)

    # Annual percentage numeric
    if 'Annual-Percentage' in df.columns:
        df['AnnualPercentage'] = df['Annual-Percentage'].apply(convert_percentage)
        df.drop('Annual-Percentage', axis=1, inplace=True)

    # Income numeric
    if 'Income' in df.columns:
        df['IncomeNum'] = df['Income'].apply(convert_income)
        df.drop('Income', axis=1, inplace=True)

    # Fill missing numeric
    num_cols = ['AnnualPercentage', 'IncomeNum']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical
    cat_cols = [c for c in df.columns if df[c].dtype == object and c != 'Outcome']
    for c in cat_cols:
        df[c] = df[c].fillna('Unknown')

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/raw/scholars.csv')
    parser.add_argument('--out', type=str, default='data/processed/')
    args = parser.parse_args()

    # Automatically detect CSV vs Excel
    if args.input.lower().endswith('.csv'):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_excel(args.input, engine='openpyxl')

    df_clean = preprocess(df)

    train, test = train_test_split(
        df_clean,
        test_size=0.2,
        random_state=42,
        stratify=df_clean['Outcome']
    )

    train.to_csv(args.out + 'train.csv', index=False)
    test.to_csv(args.out + 'test.csv', index=False)
    print('✅ Saved processed train/test to', args.out)
