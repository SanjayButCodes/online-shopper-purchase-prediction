import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from preprocess import OUT


def metrics_line(name, model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return (f"{name}: accuracy {accuracy_score(y, pred):.3f} | "
            f"precision {precision_score(y, pred):.3f} | "
            f"recall {recall_score(y, pred):.3f} | "
            f"F1 {f1_score(y, pred):.3f} | "
            f"ROC-AUC {roc_auc_score(y, prob):.3f}")


def evaluate(model, X_train, X_test, y_train, y_test):
    lines = [metrics_line("Train", model, X_train, y_train),
             metrics_line("Test", model, X_test, y_test)]

    print("\n=== Evaluation ===")
    print("\n".join(lines))

    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion matrix (test set)")
    plt.savefig(os.path.join(OUT, "confusion_matrix.png"))
    plt.close()

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(os.path.join(OUT, "roc_curve.png"))
    plt.close()