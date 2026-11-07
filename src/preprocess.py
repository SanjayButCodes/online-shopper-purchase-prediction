import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE, "data", "online_shoppers_intention.csv")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

CAT_COLS = ["Month", "VisitorType", "OperatingSystems",
            "Browser", "Region", "TrafficType"]


def load_raw():
    return pd.read_csv(DATA_FILE)


def get_data():
    df = load_raw().drop_duplicates()
    df[["Revenue", "Weekend"]] = df[["Revenue", "Weekend"]].astype(int)
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)

    y = df.pop("Revenue")
    X = df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # fit on train only so test stats don't leak
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, list(X.columns)
