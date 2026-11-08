import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocess import load_raw, OUT

CODE_COLS = ["OperatingSystems", "Browser", "Region", "TrafficType"]


def run():
    df = load_raw()
    numbers = df.drop(columns=CODE_COLS).select_dtypes(include="number")
    corr = numbers.corr()
    revenue = df["Revenue"].astype(int)

    for title, stat in [
        ("Mean", numbers.mean()),
        ("Median", numbers.median()),
        ("Standard deviation", numbers.std()),
        ("Covariance", numbers.cov()),
        ("Correlation", corr),
        ("Correlation of each column with Revenue (purchase)",
         numbers.corrwith(revenue).sort_values(ascending=False)),
    ]:
        print(f"=== {title} ===")
        print(stat, end="\n\n")

    print("=== Class balance (Revenue) ===")
    print(df["Revenue"].value_counts(normalize=True))

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Correlation heatmap (numeric columns)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "correlation_heatmap.png"))
    plt.close()
