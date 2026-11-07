import os
import numpy as np
import pandas as pd 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from preprocess import OUT

def make_model():
    return LogisticRegression(max_iter=1000, class_weight="balanced")

def train(X_train, y_train, column_names):
    model = make_model().fit(X_train, y_train)

    coefs = pd.Series(model.coef_[0], index=column_names).sort_values()
    pd.concat([coefs.head(8), coefs.tail(8)]).plot.barh(figsize=(8,6))
    plt.title("Largest coefficients")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "top_coefficients.png"))
    plt.close()

    odds = np.exp(coefs).sort_values(ascending=False)
    odds.to_csv(os.path.join(OUT, "odds_ratios.csv"), header =["odds_ratio"])

    sizes, train_scores, val_scores = learning_curve(make_model(), X_train, y_train, cv=5, scoring="f1", train_sizes=np.linspace(0.1, 1.0,5))
    plt.plot(sizes, train_scores.mean(axis=1), label="Train")
    plt.plot(sizes, val_scores.mean(axis=1), label="Validation")
    plt.xlabel("Number of training examples")
    plt.ylabel("F1 score")
    plt.legend()
    plt.savefig(os.path.join(OUT, "learning_curve.png"))
    plt.close()

    probs = model.predict_proba(X_train)[:,1]

    plt.hist(probs[y_train.values == 0], bins=30, alpha=0.6, label="No purchase")
    plt.hist(probs[y_train.values == 1], bins=30, alpha=0.6, label="Purchase")
    plt.xlabel("Predicted probability of purchase")
    plt.legend()
    plt.savefig(os.path.join(OUT, "probability_histogram.png"))
    plt.close()

    return model 
    