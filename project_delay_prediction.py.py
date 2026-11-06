
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    RocCurveDisplay
)

# --------------------------------------------------
# 1. BASIC SETTINGS
# --------------------------------------------------

SEED = 42
OUT = "outputs"

os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------
# 2. GENERATE DATASET
# --------------------------------------------------

def generate_dataset(n=2000, seed=SEED):

    rng = np.random.default_rng(seed)

    df = pd.DataFrame({

        "team_size": rng.integers(3, 40, n),

        "planned_duration_days": rng.integers(30, 540, n),

        "budget_usd": rng.integers(10000, 1000000, n),

        "requirement_changes": rng.poisson(5, n),

        "avg_team_experience_yrs": rng.uniform(1, 15, n).round(1),

        "complexity_score": rng.integers(1, 6, n),

        "stakeholders": rng.integers(1, 15, n),

        "velocity_variance_pct": rng.uniform(0, 60, n).round(1),

        "bugs_per_100_tasks": rng.uniform(0, 40, n).round(1),

        "uses_agile": rng.integers(0, 2, n),

        "uses_ai_tools": rng.integers(0, 2, n)

    })

    # Calculate project risk score

    z = (
        -3.0
        + 0.25 * df.requirement_changes
        + 0.45 * df.complexity_score
        + 0.04 * df.velocity_variance_pct
        + 0.04 * df.bugs_per_100_tasks
        + 0.08 * df.stakeholders
        + 0.002 * df.planned_duration_days
        - 0.15 * df.avg_team_experience_yrs
        - 0.6 * df.uses_agile
        - 0.5 * df.uses_ai_tools
        + rng.normal(0, 0.8, n)
    )

    # Convert risk score into probability

    prob = 1 / (
        1 + np.exp(-(z - np.median(z)) * 0.8)
    )

    # 0 = On time, 1 = Delayed

    df["delayed"] = (
        rng.random(n) < prob
    ).astype(int)

    return df


# --------------------------------------------------
# 3. DATA EXPLORATION
# --------------------------------------------------

def explore(df):

    print("\n========== DATASET INFORMATION ==========")

    print("\nDataset Shape:", df.shape)

    print("\nFirst Five Records:")

    print(df.head())

    print("\nMissing Values:")

    print(df.isnull().sum())

    print("\nDelay Distribution:")

    print(
        df["delayed"]
        .value_counts(normalize=True)
        .round(3)
    )

    # Correlation graph

    corr = (
        df.corr(numeric_only=True)["delayed"]
        .drop("delayed")
        .sort_values()
    )

    corr.plot(
        kind="barh",
        figsize=(8, 5),
        title="Correlation with Project Delay"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUT}/correlation.png",
        dpi=150
    )

    plt.close()

    print("\nCorrelation graph saved.")


# --------------------------------------------------
# 4. TRAIN AND COMPARE MODELS
# --------------------------------------------------

def train_and_compare(X_train, X_test, y_train, y_test):

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=SEED,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=SEED
        )

    }

    rows = []

    fitted = {}

    print("\n========== MODEL TRAINING ==========")

    for name, model in models.items():

        print("\nTraining:", name)

        # Train model

        model.fit(X_train, y_train)

        # Predict test data

        pred = model.predict(X_test)

        # Calculate delay probabilities

        proba = model.predict_proba(X_test)[:, 1]

        # Five-fold cross validation

        cv = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="f1",
            n_jobs=-1
        ).mean()

        # Calculate evaluation metrics

        rows.append({

            "Model": name,

            "Accuracy": accuracy_score(y_test, pred),

            "Precision": precision_score(
                y_test, pred, zero_division=0
            ),

            "Recall": recall_score(
                y_test, pred, zero_division=0
            ),

            "F1 Score": f1_score(
                y_test, pred, zero_division=0
            ),

            "ROC-AUC": roc_auc_score(y_test, proba),

            "CV F1": cv

        })

        fitted[name] = model

    # Create comparison table

    results = (
        pd.DataFrame(rows)
        .set_index("Model")
        .round(3)
    )

    print("\n========== MODEL COMPARISON ==========")

    print(results)

    results.to_csv(
        f"{OUT}/model_comparison.csv"
    )

    # Model comparison graph

    results[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Machine Learning Model Comparison")

    plt.ylabel("Score")

    plt.ylim(0, 1)

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        f"{OUT}/model_comparison.png",
        dpi=150
    )

    plt.close()

    return results, fitted


# --------------------------------------------------
# 5. EVALUATE BEST MODEL
# --------------------------------------------------

def evaluate_best(results, fitted, X_test, y_test, feature_names):

    # Select model with highest F1 score

    best_name = results["F1 Score"].idxmax()

    best = fitted[best_name]

    print("\n========== BEST MODEL ==========")

    print("Best Model:", best_name)

    pred = best.predict(X_test)

    proba = best.predict_proba(X_test)[:, 1]

    # Classification report

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            pred,
            labels=[0, 1],
            target_names=["On time", "Delayed"],
            zero_division=0
        )
    )

    # Confusion matrix

    cm = confusion_matrix(
        y_test,
        pred,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["On time", "Delayed"]
    )

    display.plot(cmap="Blues")

    plt.title(f"Confusion Matrix - {best_name}")

    plt.tight_layout()

    plt.savefig(
        f"{OUT}/confusion_matrix.png",
        dpi=150
    )

    plt.close()

    # ROC curve

    RocCurveDisplay.from_predictions(
        y_test,
        proba
    )

    plt.title(f"ROC Curve - {best_name}")

    plt.tight_layout()

    plt.savefig(
        f"{OUT}/roc_curve.png",
        dpi=150
    )

    plt.close()

    # Feature importance

    if hasattr(best, "feature_importances_"):

        importance = best.feature_importances_

    else:

        importance = np.abs(
            best.named_steps["classifier"].coef_[0]
        )

    imp = pd.Series(
        importance,
        index=feature_names
    )

    imp = imp.sort_values(ascending=False)

    print("\nTop Five Important Features:")

    print(imp.head(5).round(3))

    imp.sort_values().plot(
        kind="barh",
        figsize=(8, 6),
        title="Project Delay Risk Factors"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUT}/feature_importance.png",
        dpi=150
    )

    plt.close()

    # Save model

    joblib.dump(
        best,
        f"{OUT}/best_model.joblib"
    )

    print("\nBest model saved successfully.")

    return best


# --------------------------------------------------
# 6. PREDICT NEW PROJECT
# --------------------------------------------------

def predict_project(model, project):

    project_df = pd.DataFrame([project])

    # Predict delay probability

    prob = model.predict_proba(
        project_df
    )[0, 1]

    # Determine risk level

    if prob < 0.35:

        level = "LOW"

        advice = (
            "Project is progressing with low risk. "
            "Continue as planned and monitor weekly."
        )

    elif prob < 0.65:

        level = "MEDIUM"

        advice = (
            "Review project scope, reduce requirement "
            "changes and add buffer time."
        )

    else:

        level = "HIGH"

        advice = (
            "Take immediate action. Freeze unnecessary "
            "scope changes, assign experienced staff "
            "and re-plan project milestones."
        )

    print("\n========== PROJECT PREDICTION ==========")

    print(f"Delay Probability: {prob:.2%}")

    print("Risk Level:", level)

    print("Recommendation:", advice)

    return prob, level


# --------------------------------------------------
# 7. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\n======================================")

    print(" SOFTWARE PROJECT DELAY PREDICTION")

    print(" MACHINE LEARNING PROJECT")

    print("======================================")

    # Generate dataset

    df = generate_dataset()

    df.to_csv(
        f"{OUT}/project_data.csv",
        index=False
    )

    print("\nDataset generated successfully.")

    # Explore dataset

    explore(df)

    # Separate input and output

    X = df.drop(columns=["delayed"])

    y = df["delayed"]

    # Split training and testing data

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED

    )

    print("\nTraining Records:", len(X_train))

    print("Testing Records:", len(X_test))

    # Train and compare models

    results, fitted = train_and_compare(

        X_train,
        X_test,
        y_train,
        y_test

    )

    # Evaluate best model

    best = evaluate_best(

        results,
        fitted,
        X_test,
        y_test,
        X.columns

    )

    # New project details

    new_project = {

        "team_size": 12,

        "planned_duration_days": 180,

        "budget_usd": 250000,

        "requirement_changes": 14,

        "avg_team_experience_yrs": 3.5,

        "complexity_score": 5,

        "stakeholders": 10,

        "velocity_variance_pct": 40.0,

        "bugs_per_100_tasks": 25.0,

        "uses_agile": 0,

        "uses_ai_tools": 0

    }

    # Predict new project

    predict_project(best, new_project)

    print("\n======================================")

    print(" PROJECT EXECUTED SUCCESSFULLY")

    print(" Check the outputs folder for results.")

    print("======================================")


# --------------------------------------------------
# 8. PROGRAM EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    main()
