"""Linear Algebra for AI: appliance data and a calculated KWH proxy.

Based on the structure of the instructor's sample (parts A-H).
Run: python main.py
     python main.py --input data_1000.csv --output outputs

IMPORTANT: KWH calculated here is not measured energy consumption. In this
project's spreadsheet KWH = ElectricityBill / (ElectricityBill / MonthlyHours),
so it reproduces MonthlyHours wherever the calculation is defined. The model
is an educational proxy-prediction exercise, not a real-world kWh model.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Saves plots without requiring a display.
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 42
TEST_SIZE = 0.25
PCA_VARIANCE_THRESHOLD = 0.95
FEATURES = ["Fan", "Refrigerator", "AirConditioner", "Television", "Monitor"]
BASE_DIR = Path(__file__).resolve().parent


def fit_ols(features, target):
    """Fit ordinary least squares with an intercept; avoid explicit inversion."""
    design = np.column_stack([np.ones(len(features)), features])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients


def predict(features, coefficients):
    design = np.column_stack([np.ones(len(features)), features])
    return design @ coefficients


def evaluate(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r_squared = r2_score(actual, predicted)
    return rmse, r_squared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE_DIR / "data_1000.csv",
                        help="CSV with five appliance fields, MonthlyHours and ElectricityBill")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "outputs",
                        help="Folder for plots and test predictions")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    np.set_printoptions(precision=6, suppress=True)

    # A. Dataset and experimental design.
    df = pd.read_csv(args.input)
    required = FEATURES + ["MonthlyHours", "ElectricityBill"]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {missing_columns}")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[required].isna().any().any():
        raise ValueError("Required fields contain missing or nonnumeric values; inspect the CSV.")
    if not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError("Required fields must contain finite numeric values.")
    if len(df) < 8 or (df["MonthlyHours"] == 0).any() or (df["ElectricityBill"] == 0).any():
        raise ValueError("Need at least 8 rows and nonzero MonthlyHours and ElectricityBill to reconstruct the proxy.")

    # The exact formula used in the group's Google Sheet. It is circular as a
    # physical-energy estimate: it reconstructs MonthlyHours, not metered kWh.
    implied_rate = df["ElectricityBill"] / df["MonthlyHours"]
    y = (df["ElectricityBill"] / implied_rate).to_numpy(dtype=float)
    if "KWH" in df.columns:
        sheet_kwh = pd.to_numeric(df["KWH"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(sheet_kwh).all() or not np.allclose(sheet_kwh, y, rtol=1e-9, atol=1e-9):
            raise ValueError("CSV KWH values do not match the proxy formula; check whether the spreadsheet changed.")
    X = df[FEATURES].to_numpy(dtype=float)
    row_ids = np.arange(1, len(df) + 1)  # CSV row positions, not real household identities.
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=TEST_SIZE,
                                           random_state=RANDOM_STATE)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    n, p = X_train.shape
    print("PART A — DATASET AND SPLIT")
    print("Source CSV:", args.input, "| rows:", len(df))
    print("Target: calculated proxy = bill / (bill / MonthlyHours); NOT measured kWh")
    print("Feature order:", FEATURES)
    print("Train rows:", n, "| test rows:", len(test_idx), "| random_state:", RANDOM_STATE)
    print("Maximum proxy vs MonthlyHours difference:", np.max(np.abs(y - df["MonthlyHours"].to_numpy())))

    # B. Matrix representation.
    raw_gram = X_train.T @ X_train
    print("\nPART B — MATRICES")
    print("X_train shape:", X_train.shape, "| X_test shape:", X_test.shape)
    print("X_train transpose shape:", X_train.T.shape, "| Gram shape:", raw_gram.shape)
    print("Raw Gram matrix:\n", raw_gram)

    # C. Vector spaces and feature dependence.
    print("\nPART C — VECTOR SPACES")
    print("Feature-space dimension:", p)
    print("Rank of raw training matrix:", np.linalg.matrix_rank(X_train))

    # D. Standardize with training parameters only.
    scaler = StandardScaler().fit(X_train)
    if np.any(np.var(X_train, axis=0) == 0):
        raise ValueError("At least one training feature is constant; scaling matrix is not invertible.")
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)
    M = np.diag(1.0 / scaler.scale_)
    print("\nPART D — LINEAR TRANSFORMATION")
    print("Training means:", scaler.mean_)
    print("Training population standard deviations:", scaler.scale_)
    print("Scaling matrix M:\n", M, "\ndet(M):", np.linalg.det(M))
    print("First test raw:", X_test[0], "| standardized:", X_test_std[0])

    # E. Determinant and conditioning of standardized Gram matrix.
    G = X_train_std.T @ X_train_std
    print("\nPART E — DETERMINANTS AND CONDITIONING")
    print("Standardized Gram matrix:\n", G)
    print("det(G):", np.linalg.det(G), "| condition number:", np.linalg.cond(G),
          "| rank:", np.linalg.matrix_rank(G))

    # F. Eigensystem for the sample covariance of standardized training data.
    C = np.cov(X_train_std, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    # Eigenvector direction is arbitrary; choose a consistent sign for reporting.
    for j in range(p):
        if eigenvectors[np.argmax(np.abs(eigenvectors[:, j])), j] < 0:
            eigenvectors[:, j] *= -1
    variance_ratio = eigenvalues / eigenvalues.sum()
    cumulative_variance = np.cumsum(variance_ratio)
    print("\nPART F — EIGENVALUES AND EIGENVECTORS")
    print("Covariance matrix:\n", C)
    print("Eigenvalues:", eigenvalues, "| explained variance:", variance_ratio)
    print("Cumulative variance:", cumulative_variance)
    print("PC1 loadings:", eigenvectors[:, 0])
    print("Leading eigenpair residual:", np.linalg.norm(C @ eigenvectors[:, 0] - eigenvalues[0] * eigenvectors[:, 0]))

    # G. Diagonalization and projection; all components may be necessary.
    reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    k = int(np.searchsorted(cumulative_variance, PCA_VARIANCE_THRESHOLD) + 1)
    k = min(k, p)
    X_train_pca = X_train_std @ eigenvectors[:, :k]
    X_test_pca = X_test_std @ eigenvectors[:, :k]
    print("\nPART G — DIAGONALIZATION AND PCA")
    print("Reconstruction error:", np.linalg.norm(C - reconstructed, ord="fro"))
    print(f"Components for >= {PCA_VARIANCE_THRESHOLD:.0%} variance: {k}; retained: {cumulative_variance[k-1]:.4%}")
    print("PCA train/test shapes:", X_train_pca.shape, X_test_pca.shape)

    # H. OLS evaluation on the same held-out records for both models.
    beta_full = fit_ols(X_train_std, y_train)
    beta_pca = fit_ols(X_train_pca, y_train)
    pred_full = predict(X_test_std, beta_full)
    pred_pca = predict(X_test_pca, beta_pca)
    rmse_full, r2_full = evaluate(y_test, pred_full)
    rmse_pca, r2_pca = evaluate(y_test, pred_pca)
    print("\nPART H — MODEL EVALUATION (PROXY ONLY)")
    print(f"Full standardized OLS: RMSE={rmse_full:.4f}, R²={r2_full:.4f}")
    print(f"PCA OLS (k={k}): RMSE={rmse_pca:.4f}, R²={r2_pca:.4f}")
    print("Full-model intercept and coefficients:", beta_full)
    print("First test row ID:", row_ids[test_idx[0]], "| actual:", y_test[0], "| prediction:", pred_full[0])
    if k == p:
        print("All components retained: PCA is a rotation, not dimension reduction.")
    pd.DataFrame({"source_row_id": row_ids[test_idx], "actual_proxy": y_test,
                  "full_prediction": pred_full, "pca_prediction": pred_pca}).to_csv(
                      args.output / "test_predictions.csv", index=False)

    # The three visualizations in the instructor's sample; written as PNG files.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, p + 1), variance_ratio * 100, color="#3478a5")
    ax.plot(range(1, p + 1), cumulative_variance * 100, marker="o", color="#dc7641", label="Cumulative")
    ax.axhline(95, linestyle="--", color="gray", linewidth=1)
    ax.set(xlabel="Principal component", ylabel="Explained variance (%)", title="Training PCA explained variance")
    ax.set_xticks(range(1, p + 1)); ax.legend(); fig.tight_layout()
    fig.savefig(args.output / "scree_plot.png", dpi=180); plt.close(fig)

    # Plot first two components even if k=1, using the full eigensystem.
    projection = X_train_std @ eigenvectors[:, :2]
    fig, ax = plt.subplots(figsize=(6, 4))
    points = ax.scatter(projection[:, 0], projection[:, 1], c=y_train, cmap="viridis", alpha=0.7, s=14)
    ax.set(xlabel="PC1", ylabel="PC2", title="Training data projected onto PC1 and PC2")
    fig.colorbar(points, ax=ax, label="Calculated KWH proxy (not measured kWh)")
    fig.tight_layout(); fig.savefig(args.output / "pca_projection.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, pred_full, alpha=0.6, s=16)
    low, high = min(y_test.min(), pred_full.min()), max(y_test.max(), pred_full.max())
    ax.plot([low, high], [low, high], "r--", label="Perfect agreement")
    ax.set(xlabel="Actual calculated proxy", ylabel="Predicted proxy", title="Full OLS: predicted vs actual proxy")
    ax.legend(); fig.tight_layout(); fig.savefig(args.output / "predicted_vs_actual.png", dpi=180); plt.close(fig)
    print("\nSaved figures and test_predictions.csv to:", args.output.resolve())


if __name__ == "__main__":
    main()
