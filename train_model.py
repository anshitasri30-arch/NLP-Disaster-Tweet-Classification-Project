"""Train and evaluate disaster-tweet classification models.
Put the real dataset in this folder as twitter_disaster.xlsx or twitter_disaster.csv.
Expected columns: text and target (target: 0 = non-disaster, 1 = disaster).
"""
from pathlib import Path
import re, json, warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_curve,
                             precision_recall_curve, auc, average_precision_score)
from sklearn.dummy import DummyClassifier
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

def locate_dataset():
    candidates = [
        ROOT / "twitter_disaster.xlsx", ROOT / "twitter_disaster.xlsx",
        ROOT / "twitter_disaster.csv", ROOT / "train.csv", ROOT / "dataset.xlsx",
        ROOT / "dataset.csv"
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Dataset not found. Place the REAL Excel/CSV file in this project folder as "
        "'twitter_disaster.xlsx' or 'twitter_disaster.csv'. The provided .mhtml webpage "
        "snapshot is not a usable Excel dataset. In Google Sheets choose File > Download > "
        "Microsoft Excel (.xlsx) or Comma-separated values (.csv)."
    )

def load_data(path):
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Handle common alternate label names.
    text_col = next((c for c in ("text", "tweet", "content", "message") if c in df.columns), None)
    target_col = next((c for c in ("target", "label", "class", "disaster", "is_disaster") if c in df.columns), None)
    if text_col is None or target_col is None:
        raise ValueError(
            f"Could not identify text/label columns. Found columns: {list(df.columns)}. "
            "Expected a text column named 'text' and a label column named 'target' (0/1)."
        )
    df = df[[text_col, target_col]].rename(columns={text_col: "text", target_col: "target"})
    df = df.dropna(subset=["text", "target"]).copy()
    df["text"] = df["text"].astype(str)
    # Labels should be numeric 0/1; accept common text labels too.
    if not pd.api.types.is_numeric_dtype(df["target"]):
        mapping = {
            "0": 0, "non-disaster": 0, "non disaster": 0, "not disaster": 0,
            "false": 0, "no": 0, "1": 1, "disaster": 1, "real disaster": 1,
            "true": 1, "yes": 1
        }
        df["target"] = df["target"].astype(str).str.strip().str.lower().map(mapping)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)
    df = df[df["target"].isin([0, 1])]
    df = df.drop_duplicates(subset=["text"])
    if df["target"].nunique() < 2:
        raise ValueError("The dataset must contain both target classes 0 and 1.")
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9#'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    path = locate_dataset()
    df = load_data(path)
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    if len(df) < 10:
        raise ValueError(f"Only {len(df)} usable rows found; more labeled examples are needed.")

    # Data exploration outputs
    summary = {
        "dataset_file": path.name,
        "rows_after_cleaning": int(len(df)),
        "class_counts": {str(int(k)): int(v) for k, v in df["target"].value_counts().sort_index().items()},
        "columns_used": ["text", "target"],
        "label_definition": {"0": "Non-disaster", "1": "Disaster"}
    }
    (REPORT_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(7, 4))
    counts = df["target"].value_counts().reindex([0, 1], fill_value=0)
    sns.barplot(x=["Non-disaster (0)", "Disaster (1)"], y=counts.values)
    plt.title("Distribution of Tweet Classes")
    plt.ylabel("Number of tweets")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "class_distribution.png", dpi=160)
    plt.close()

    # Frequent words for disaster tweets (simple transparent analysis)
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(stop_words="english", max_features=20, ngram_range=(1, 2))
    matrix = cv.fit_transform(df.loc[df.target == 1, "clean_text"])
    freqs = matrix.sum(axis=0).A1
    vocab = cv.get_feature_names_out()
    word_df = pd.DataFrame({"term": vocab, "frequency": freqs}).sort_values("frequency", ascending=False)
    word_df.to_csv(REPORT_DIR / "disaster_keyword_frequency.csv", index=False)
    if len(word_df):
        plt.figure(figsize=(9, 6))
        top = word_df.head(15).sort_values("frequency")
        plt.barh(top["term"], top["frequency"])
        plt.title("Frequent Terms in Disaster Tweets")
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "disaster_keywords.png", dpi=160)
        plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )
    cv_splits = min(5, int(y_train.value_counts().min()))
    if cv_splits < 2:
        raise ValueError("Not enough examples per class for cross-validation. Add more labeled data.")

    models = {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=12000, min_df=2)),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                           random_state=42, n_jobs=-1))
        ])
    }
    results = []
    best_name, best_model, best_f1 = None, None, -1
    for name, pipeline in models.items():
        # Cross-validation scores are used for model comparison.
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(
            n_splits=cv_splits, shuffle=True, random_state=42), scoring="f1", n_jobs=None)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        acc = accuracy_score(y_test, pred)
        results.append({
            "model": name, "cv_f1_mean": float(scores.mean()), "cv_f1_std": float(scores.std()),
            "accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1_score": float(f1)
        })
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, pipeline, f1

    metrics = {"selected_model": best_name, "test_rows": int(len(y_test)), "models": results}
    (REPORT_DIR / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-disaster", "Disaster"], yticklabels=["Non-disaster", "Disaster"])
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    if hasattr(best_model, "predict_proba"):
        prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
        plt.savefig(REPORT_DIR / "roc_curve.png", dpi=160); plt.close()
        pr, rc, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        plt.figure(figsize=(5, 4))
        plt.plot(rc, pr, label=f"Average precision = {ap:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision–Recall Curve"); plt.legend(); plt.tight_layout()
        plt.savefig(REPORT_DIR / "precision_recall_curve.png", dpi=160); plt.close()
        metrics["roc_auc"] = float(roc_auc)
        metrics["average_precision"] = float(ap)
        (REPORT_DIR / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    (REPORT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, pred, target_names=["Non-disaster", "Disaster"], zero_division=0),
        encoding="utf-8"
    )
    joblib.dump(best_model, MODEL_DIR / "disaster_tweet_model.joblib")
    (MODEL_DIR / "model_info.json").write_text(json.dumps({
        "model_name": best_name, "text_column": "text", "label_column": "target",
        "labels": {"0": "Non-disaster", "1": "Disaster"}
    }, indent=2), encoding="utf-8")
    print(f"Training completed using: {path.name}")
    print(f"Rows used: {len(df)}")
    print(f"Selected model: {best_name}")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"Outputs saved in: {REPORT_DIR}")
    print(f"Model saved in: {MODEL_DIR / 'disaster_tweet_model.joblib'}")

if __name__ == "__main__":
    main()
