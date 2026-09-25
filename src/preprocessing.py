import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = "data/raw/phishing_dataset.csv"
X_TRAIN_PATH = "data/processed/X_train.csv"
X_TEST_PATH = "data/processed/X_test.csv"
Y_TRAIN_PATH = "data/processed/y_train.csv"
Y_TEST_PATH = "data/processed/y_test.csv"


def clean_value(value):
    """Convert ARFF byte-string values into integers."""
    if isinstance(value, str):
        value = value.strip()

        if value.startswith("b'") and value.endswith("'"):
            value = value[2:-1]

        if value in {"-1", "0", "1"}:
            return int(value)

    return value


def preprocess_data():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    print(f"Original shape: {df.shape}")

    # Convert all byte-string values to numeric values
    for column in df.columns:
        df[column] = df[column].apply(clean_value)

    # Separate features and target
    X = df.drop(columns=["Result"])
    y = df["Result"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Save processed datasets
    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    preprocess_data()