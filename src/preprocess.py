import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE, "data", "online_shoppers_intention.csv")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

SKEWED_COLS = ["Administrative", "Administrative_Duration",
               "Informational", "Informational_Duration",
               "ProductRelated", "ProductRelated_Duration", "PageValues"]

KEEP_TOP = {"OperatingSystems": 3, "Browser": 3, "TrafficType": 3}
CAT_COLS = ["Month", "VisitorType", *KEEP_TOP]


def load_raw():
    return pd.read_csv(DATA_FILE)


def get_data():
    df = load_raw().drop_duplicates()
    df[["Revenue", "Weekend"]] = df[["Revenue", "Weekend"]].astype(int)
    df = df.drop(columns=["Region", "BounceRates"])

    y = df.pop("Revenue")
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = X_train.copy(), X_test.copy()

    # top codes come from train only so test doesn't leak
    for col, n in KEEP_TOP.items():
        common = X_train[col].value_counts().head(n).index
        for part in (X_train, X_test):
            part[col] = part[col].where(part[col].isin(common), "Other").astype(str)

    for part in (X_train, X_test):
        part[SKEWED_COLS] = np.log1p(part[SKEWED_COLS])

    X_train = pd.get_dummies(X_train, columns=CAT_COLS, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=CAT_COLS, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    column_names = list(X_train.columns)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.astype(float))
    X_test = scaler.transform(X_test.astype(float))

    return X_train, X_test, y_train, y_test, column_names
