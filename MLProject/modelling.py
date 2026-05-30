import os, sys, argparse, json, shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)
from mlflow.models.signature import infer_signature

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int,   default=100)
parser.add_argument("--max_depth",    type=int,   default=10)
parser.add_argument("--random_state", type=int,   default=42)
parser.add_argument("--test_size",    type=float, default=0.2)
args = parser.parse_args()

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "https://dagshub.com/aadnauj/Eksperimen_SML_Juanda-Harefa.mlflow")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Churn_Workflow_CI")

print(f"n_estimators={args.n_estimators}, max_depth={args.max_depth}")

# Cari CSV dataset raw
csv_path = None
for root, dirs, files in os.walk("."):
    for f in files:
        if f.endswith(".csv") and "churn" in f.lower():
            csv_path = os.path.join(root, f)
            break
    if csv_path:
        break

if not csv_path:
    raise FileNotFoundError("Dataset CSV tidak ditemukan!")

print(f"Memuat: {csv_path}")
data = pd.read_csv(csv_path)
data = data.drop(columns=[c for c in ["RowNumber","CustomerId","Surname"] if c in data.columns])
for col in ["Geography","Gender"]:
    if col in data.columns:
        data[col] = LabelEncoder().fit_transform(data[col])

X = data.drop(columns=["Exited"])
y = data["Exited"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test  = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns)
print(f"X_train={X_train.shape}, X_test={X_test.shape}")

os.makedirs("tmp_artifacts", exist_ok=True)

with mlflow.start_run(run_name=f"RF_n{args.n_estimators}_d{args.max_depth}"):
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth",    args.max_depth)
    mlflow.log_param("random_state", args.random_state)
    mlflow.log_param("test_size",    args.test_size)
    mlflow.log_metric("accuracy",       acc)
    mlflow.log_metric("precision",      prec)
    mlflow.log_metric("recall",         rec)
    mlflow.log_metric("f1_score",       f1)
    mlflow.log_metric("roc_auc",        roc_auc)
    mlflow.log_metric("true_positive",  int(tp))
    mlflow.log_metric("false_positive", int(fp))
    mlflow.log_metric("false_negative", int(fn))
    mlflow.log_metric("true_negative",  int(tn))
    mlflow.set_tag("model",  "RandomForestClassifier")
    mlflow.set_tag("source", "Workflow-CI")

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("tmp_artifacts/confusion_matrix.png"); plt.close()
    mlflow.log_artifact("tmp_artifacts/confusion_matrix.png", "plots")

    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    ax.plot([0,1],[0,1],"--", color="gray")
    ax.set_title("ROC Curve"); ax.legend()
    plt.tight_layout()
    plt.savefig("tmp_artifacts/roc_curve.png"); plt.close()
    mlflow.log_artifact("tmp_artifacts/roc_curve.png", "plots")

    report = classification_report(y_test, y_pred, output_dict=True)
    with open("tmp_artifacts/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact("tmp_artifacts/classification_report.json", "reports")

    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model",
                             signature=signature, registered_model_name="RandomForest_WorkflowCI")

    print(f"Acc={acc:.4f} | F1={f1:.4f} | AUC={roc_auc:.4f}")

shutil.rmtree("tmp_artifacts", ignore_errors=True)
print("Selesai!")
