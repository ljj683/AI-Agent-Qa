import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from src.config import LABELS_DIR, STORAGE_DIR, REPORTS_DIR


def load_data(path=LABELS_DIR / "text_classification.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].astype(str).str.len() >= 4]
    return df


def train_and_evaluate():
    (STORAGE_DIR / "ml_models").mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    models = {
        "NaiveBayes": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer(analyzer="char")),
                ("clf", MultinomialNB())
            ]),
            "params": {
                "tfidf__max_features": [1000, 3000, 5000],
                "tfidf__ngram_range": [(1, 2), (2, 4)],
                "clf__alpha": [0.1, 0.5, 1.0]
            }
        },
        "LogisticRegression": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer(analyzer="char")),
                ("clf", LogisticRegression(max_iter=1000))
            ]),
            "params": {
                "tfidf__max_features": [1000, 3000, 5000],
                "tfidf__ngram_range": [(1, 2), (2, 4)],
                "clf__C": [0.1, 1, 10]
            }
        },
        "MLP_NeuralNetwork": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer(analyzer="char")),
                ("clf", MLPClassifier(max_iter=700, random_state=42, early_stopping=True))
            ]),
            "params": {
                "tfidf__max_features": [1000, 3000],
                "tfidf__ngram_range": [(1, 2), (2, 4)],
                "clf__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "clf__alpha": [0.0001, 0.001]
            }
        }
    }

    results = []
    best_model = None
    best_f1 = -1
    best_name = ""

    for name, item in models.items():
        print(f"\n开始训练模型：{name}")
        grid = GridSearchCV(item["pipeline"], item["params"], cv=3, scoring="f1_macro", n_jobs=-1)
        grid.fit(X_train, y_train)
        pred = grid.predict(X_test)

        acc = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average="macro", zero_division=0)
        recall = recall_score(y_test, pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, pred, average="macro", zero_division=0)

        result = {
            "model": name,
            "accuracy": acc,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "best_params": str(grid.best_params_)
        }
        results.append(result)
        print(classification_report(y_test, pred, zero_division=0))
        joblib.dump(grid.best_estimator_, STORAGE_DIR / "ml_models" / f"{name}.pkl")

        if f1 > best_f1:
            best_f1 = f1
            best_model = grid.best_estimator_
            best_name = name

    joblib.dump(best_model, STORAGE_DIR / "ml_models" / "best_text_classifier.pkl")
    pd.DataFrame(results).to_csv(REPORTS_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    (REPORTS_DIR / "best_model.txt").write_text(f"best_model={best_name}\nf1_macro={best_f1}\n", encoding="utf-8")
    print("\n模型训练完成，最优模型：", best_name)
    print(pd.DataFrame(results))


if __name__ == "__main__":
    train_and_evaluate()
