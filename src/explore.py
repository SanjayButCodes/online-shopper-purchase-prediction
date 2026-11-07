import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocess import load_raw, OUT


def run():
    df = load_raw()
    numbers = df.select_dtypes(include="number")
    corr = numbers.corr()

    for title, stat in [("Mean", numbers.mean()),
                        ("Median", numbers.median()),
                        ("Standard deviation", numbers.std()),
                        ("Covariance", numbers.cov()),
                        ("Correlation", corr)]:
        print(f"=== {title} ===")
        print(stat, end="\n\n")

    print("=== Class balance (Revenue) ===")
    print(df["Revenue"].value_counts(normalize=True))

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "correlation_heatmap.png"))
    plt.close()
