"""
Interpretable Machine Learning based Intrusion Detection
using XGBoost + Adversarial Training + SHAP   (v2 - robust loading)

- If KDDTrain+.arff is found  -> train on it, test on KDDTest+.arff
- If it is NOT found          -> split KDDTest+.arff 75/25 (with a warning)
- Only continuous features are perturbed by the attack
"""

# ---------------- 1. IMPORTS ----------------
import os
import re
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

try:
    import shap
except Exception as e:                     # SHAP is optional for the core results
    shap = None
    print(f"WARNING: could not import shap ({e}). SHAP step will be skipped.")

# ---------------- 2. SETTINGS ----------------
DATA_DIR = r"C:\Users\sonik\OneDrive\Desktop"
TRAIN_PATH = os.path.join(DATA_DIR, "KDDTrain+.arff")
TEST_PATH = os.path.join(DATA_DIR, "KDDTest+.arff")

NORMAL_LABEL = "normal"
RANDOM_STATE = 42
EPSILON = 0.30              # max change per feature (in standard deviations)
ATTACK_ITERATIONS = 60      # random-search tries per sample
ADV_TRAIN_ROUNDS = 2
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_STATE)


# ---------------- 3. LOAD DATA ----------------
def read_arff(path):
    """Read an .arff file into a DataFrame (also handles plain CSV-like data)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    names, data_start = [], 0
    for i, line in enumerate(lines):
        text = line.strip()
        lower = text.lower()
        if lower.startswith("@attribute"):
            m = re.match(r"@attribute\s+('([^']*)'|\"([^\"]*)\"|(\S+))", text,
                         flags=re.IGNORECASE)
            names.append(m.group(2) or m.group(3) or m.group(4))
        elif lower.startswith("@data"):
            data_start = i + 1
            break
    rows = [l for l in lines[data_start:] if l.strip() and not l.startswith("%")]
    return pd.read_csv(StringIO("".join(rows)), header=None, names=names,
                       quotechar="'", skipinitialspace=True)


def find_label_column(df):
    for c in ("class", "label", "Class", "Label"):
        if c in df.columns:
            return c
    return df.columns[-1]          # fall back to the last column


def split_xy(df, label_col):
    label = df[label_col].astype(str).str.strip().str.strip(".").str.lower()
    y = (label != NORMAL_LABEL).astype(int).reset_index(drop=True)
    X = df.drop(columns=[label_col]).reset_index(drop=True)
    return X, y


def load_data():
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Cannot find test file:\n  {TEST_PATH}\nFix DATA_DIR / TEST_PATH at the top.")

    test_df = read_arff(TEST_PATH)
    label_col = find_label_column(test_df)
    X_te_raw, y_te = split_xy(test_df, label_col)

    if os.path.exists(TRAIN_PATH):
        print(f"Using separate train file: {TRAIN_PATH}")
        train_df = read_arff(TRAIN_PATH)
        X_tr_raw, y_tr = split_xy(train_df, find_label_column(train_df))
    else:
        print("\n*** WARNING: KDDTrain+.arff not found. Splitting KDDTest+ 75/25. ***")
        print("*** Results will be optimistic. Put KDDTrain+.arff next to the test file. ***\n")
        X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
            X_te_raw, y_te, test_size=0.25, stratify=y_te,
            random_state=RANDOM_STATE)
        X_tr_raw = X_tr_raw.reset_index(drop=True)
        X_te_raw = X_te_raw.reset_index(drop=True)
        y_tr = y_tr.reset_index(drop=True)
        y_te = y_te.reset_index(drop=True)

    # Continuous columns = numeric with >2 distinct values (decided on train only)
    numeric_cols = X_tr_raw.select_dtypes(include=[np.number]).columns
    continuous_cols = [c for c in numeric_cols if X_tr_raw[c].nunique() > 2]

    X_tr = pd.get_dummies(X_tr_raw, dtype=float)
    X_te = pd.get_dummies(X_te_raw, dtype=float)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)

    # XGBoost dislikes [ ] < in feature names
    clean = {c: re.sub(r"[\[\]<>]", "_", str(c)) for c in X_tr.columns}
    X_tr = X_tr.rename(columns=clean)
    X_te = X_te.rename(columns=clean)
    continuous_cols = [clean[c] for c in continuous_cols]

    for d in (X_tr, X_te):
        d.replace([np.inf, -np.inf], np.nan, inplace=True)
        d.fillna(0, inplace=True)
    return X_tr.astype(float), y_tr, X_te.astype(float), y_te, continuous_cols


X_train, y_train, X_test, y_test, continuous_cols = load_data()
print(f"Train: {X_train.shape[0]} rows, {X_train.shape[1]} features, "
      f"attack ratio = {y_train.mean():.2%}")
print(f"Test : {X_test.shape[0]} rows, attack ratio = {y_test.mean():.2%}")
print(f"Continuous (perturbable) features: {len(continuous_cols)}")

# ---------------- 4. SCALE (continuous columns only) ----------------
scaler = StandardScaler().fit(X_train[continuous_cols])
X_train[continuous_cols] = scaler.transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

FEATURES = list(X_train.columns)
cont_set = set(continuous_cols)
cont_mask = np.array([c in cont_set for c in FEATURES])
train_min = X_train.values.min(axis=0)
train_max = X_train.values.max(axis=0)


# ---------------- 5. MODEL + EVALUATION ----------------
def build_model():
    return XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)


def evaluate(model, X_eval, y_eval, title):
    pred = model.predict(X_eval)
    res = {
        "accuracy": accuracy_score(y_eval, pred),
        "precision": precision_score(y_eval, pred, zero_division=0),
        "recall": recall_score(y_eval, pred, zero_division=0),
        "f1": f1_score(y_eval, pred, zero_division=0),
    }
    print(f"\n--- {title} ---")
    print({k: round(v, 4) for k, v in res.items()})
    return res


# ---------------- 6. BASELINE MODEL ----------------
baseline = build_model()
baseline.fit(X_train, y_train)
print("\nBaseline model - classification report (clean test data):")
print(classification_report(y_test, baseline.predict(X_test),
                            target_names=["Normal", "Attack"], zero_division=0))
res_base_clean = evaluate(baseline, X_test, y_test, "Baseline | clean test")


# ---------------- 7. ADVERSARIAL ATTACK (black-box random search) ----------------
def adversarial_attack(model, X_data, y_data, epsilon=EPSILON,
                       iterations=ATTACK_ITERATIONS):
    """Make ATTACK rows look normal by perturbing continuous features only."""
    X_all = X_data.values.astype(float).copy()
    attack_idx = np.where(np.asarray(y_data) == 1)[0]
    if len(attack_idx) == 0:
        return pd.DataFrame(X_all, columns=X_data.columns)

    X_orig = X_all[attack_idx]
    X_adv = X_orig.copy()
    lower = np.where(cont_mask, np.maximum(X_orig - epsilon, train_min), X_orig)
    upper = np.where(cont_mask, np.minimum(X_orig + epsilon, train_max), X_orig)

    best_score = model.predict_proba(pd.DataFrame(X_adv, columns=X_data.columns))[:, 1]
    for _ in range(iterations):
        noise = rng.uniform(-epsilon, epsilon, X_adv.shape) * cont_mask
        candidate = np.clip(X_adv + noise, lower, upper)
        score = model.predict_proba(
            pd.DataFrame(candidate, columns=X_data.columns))[:, 1]
        better = score < best_score
        X_adv[better] = candidate[better]
        best_score[better] = score[better]

    X_all[attack_idx] = X_adv
    return pd.DataFrame(X_all, columns=X_data.columns)


print("\nGenerating adversarial test samples against the baseline model ...")
X_test_adv = adversarial_attack(baseline, X_test, y_test)
res_base_adv = evaluate(baseline, X_test_adv, y_test, "Baseline | adversarial test")


# ---------------- 8. ADVERSARIAL TRAINING ----------------
robust = baseline
X_aug, y_aug = X_train.copy(), y_train.copy()
attack_rows = (y_train.values == 1)
for round_no in range(1, ADV_TRAIN_ROUNDS + 1):
    print(f"\nAdversarial training round {round_no} ...")
    X_train_adv = adversarial_attack(robust, X_train, y_train)
    X_aug = pd.concat([X_aug, X_train_adv.loc[attack_rows]], ignore_index=True)
    y_aug = pd.concat([y_aug, y_train.loc[attack_rows]], ignore_index=True)
    robust = build_model()
    robust.fit(X_aug, y_aug)

res_rob_clean = evaluate(robust, X_test, y_test, "Robust | clean test")
X_test_adv_rob = adversarial_attack(robust, X_test, y_test)
res_rob_adv = evaluate(robust, X_test_adv_rob, y_test, "Robust | adversarial test")


# ---------------- 9. RESULTS TABLE + CHART ----------------
results = pd.DataFrame({
    "Baseline (clean)": res_base_clean,
    "Baseline (adversarial)": res_base_adv,
    "Robust (clean)": res_rob_clean,
    "Robust (adversarial)": res_rob_adv,
}).T.round(4)
print("\n===== SUMMARY =====")
print(results)
results.to_csv(os.path.join(OUTPUT_DIR, "results_summary.csv"))

results[["accuracy", "precision", "recall", "f1"]].plot(
    kind="bar", figsize=(10, 5), ylim=(0, 1.05))
plt.title("Baseline vs Adversarially Trained XGBoost")
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
plt.close()


# ---------------- 10. CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, robust.predict(X_test_adv_rob), labels=[0, 1])
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Normal", "Attack"]); ax.set_yticklabels(["Normal", "Attack"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Robust model under attack")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_robust_adv.png"), dpi=150)
plt.close()


# ---------------- 11. SHAP EXPLAINABILITY ----------------
def shap_report(model, name, sample_size=500):
    sample = X_test.sample(min(sample_size, len(X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):               # old SHAP: list per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:                       # (n, features, classes)
        values = values[:, :, 1]

    plt.figure()
    shap.summary_plot(values, sample, show=False)
    plt.title(f"SHAP summary - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"shap_summary_{name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(values, sample, plot_type="bar", show=False)
    plt.title(f"Mean |SHAP| - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"shap_bar_{name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # Waterfall for one predicted attack (separate try: it is the fussiest plot)
    try:
        preds = model.predict(sample)
        idx = int(np.where(preds == 1)[0][0]) if (preds == 1).any() else 0
        base_val = explainer.expected_value
        base_val = float(np.ravel(base_val)[-1])
        exp = shap.Explanation(values=values[idx], base_values=base_val,
                               data=sample.iloc[idx].values,
                               feature_names=list(sample.columns))
        plt.figure()
        shap.plots.waterfall(exp, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"shap_waterfall_{name}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        plt.close("all")
        print(f"(waterfall plot skipped for {name}: {e})")

    return pd.Series(np.abs(values).mean(axis=0), index=FEATURES
                     ).sort_values(ascending=False)


if shap is not None:
    try:
        imp_base = shap_report(baseline, "baseline")
        imp_rob = shap_report(robust, "robust")
        top = pd.DataFrame({"baseline": imp_base, "robust": imp_rob}).sort_values(
            "robust", ascending=False).head(10).round(4)
        print("\nTop 10 features by mean |SHAP| value:")
        print(top)
        top.to_csv(os.path.join(OUTPUT_DIR, "top_features_shap.csv"))
    except Exception as e:
        plt.close("all")
        print(f"\nSHAP step failed: {e}")
        print("Try:  pip install -U shap   (or)   pip install \"xgboost<3\"")

print(f"\nDone. All plots and tables are saved in the '{OUTPUT_DIR}' folder.")