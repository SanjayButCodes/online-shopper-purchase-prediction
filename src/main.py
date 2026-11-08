import explore
import train
import evaluate
from preprocess import get_data

explore.run()
X_train, X_test, y_train, y_test, columns = get_data()
print(f"\nFeatures after preprocessing: {len(columns)}")
model = train.train(X_train, y_train, columns)
evaluate.evaluate(model, X_train, X_test, y_train, y_test)
print("\nDone. Results saved in outputs/")
