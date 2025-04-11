import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
np.random.seed(0)

def generate_synthetic_data(return_X_y=True):
    """Generate synthetic student-teacher data for regression analysis."""
    # Set parameters for data generation
    n_samples = 1000
    n_features = 20
    
    # Generate random feature matrix
    X = np.random.randn(n_samples, n_features)
    
    # Create true coefficients (teacher model)
    true_coef = np.random.randn(n_features)
    
    # Generate target values with some noise
    noise = np.random.randn(n_samples) * 0.5
    y = X @ true_coef + noise
    
    if return_X_y:
        return X, y
    else:
        return {'data': X, 'target': y}

def save_plot_with_multiple_extensions(plot_dir, plot_title):
    """Save the current plot in multiple file formats."""
    for extension in ['png', 'pdf', 'svg']:
        plt.savefig(
            os.path.join(plot_dir, f"{plot_title}.{extension}"),
            bbox_inches="tight",
            dpi=300
        )

# Create results directory
results_dir = "results/student_teacher_analysis"
os.makedirs(results_dir, exist_ok=True)

# Set parameters for the analysis
singular_value_cutoffs = np.logspace(-3, 0, 7)
num_repeats = 20  # Reduced from 50 for efficiency

# Get the data
dataset_name = "Student-Teacher"
X, y = generate_synthetic_data(return_X_y=True)

# Calculate the ideal linear relationship for ablation
beta_ideal = np.linalg.inv(X.T @ X) @ X.T @ y

# Initialize data collection
dataset_loss_unablated_df = []
dataset_loss_no_small_singular_values_df = []
dataset_loss_no_residuals_in_ideal_fit_df = []
dataset_loss_test_features_in_training_feature_subspace_df = []

# Run experiments
for repeat_idx in range(num_repeats):
    subset_sizes = np.arange(1, 40, 1)
    for subset_size in subset_sizes:
        print(f"Repeat: {repeat_idx}, Subset size: {subset_size}")
        
        # Split the data into training/testing sets
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
            X, y, np.arange(X.shape[0]),
            random_state=repeat_idx,
            test_size=X.shape[0] - subset_size,
            shuffle=True
        )
        
        # Unablated linear fit using SVD
        U, S, Vt = np.linalg.svd(X_train, full_matrices=False, compute_uv=True)
        min_singular_value = np.min(S[S > 0.0])
        S_inverted = 1.0 / S
        S_inverted[S_inverted == np.inf] = 0.0
        beta_hat_unablated = Vt.T @ np.diag(S_inverted) @ U.T @ y_train
        
        # Calculate MSE for unablated model
        y_train_pred = X_train @ beta_hat_unablated
        train_mse_unablated = mean_squared_error(y_train, y_train_pred)
        y_test_pred = X_test @ beta_hat_unablated
        test_mse_unablated = mean_squared_error(y_test, y_test_pred)
        
        # Calculate projection for test bias
        X_hat_test = X_test @ X_train.T @ np.linalg.pinv(X_train @ X_train.T) @ X_train
        X_test_diff = X_hat_test - X_test
        X_test_diff_inner_beta_ideal = np.mean(X_test_diff @ beta_ideal)
        
        # Store unablated results
        dataset_loss_unablated_df.append({
            "Dataset": dataset_name,
            "Subset Size": subset_size,
            "Num Parameters": X.shape[1],
            "Train MSE": train_mse_unablated,
            "Test MSE": test_mse_unablated,
            "Repeat Index": repeat_idx,
            "Test Bias Squared": np.square(X_test_diff_inner_beta_ideal),
            "Smallest Non-Zero Singular Value": min_singular_value
        })
        
        # Ablation 1: No small singular values
        for cutoff in singular_value_cutoffs:
            S_cutoff = np.copy(S)
            S_cutoff[S_cutoff < cutoff] = 0.0
            inverted_S_cutoff = 1.0 / S_cutoff
            inverted_S_cutoff[inverted_S_cutoff == np.inf] = 0.0
            
            beta_hat_cutoff = Vt.T @ np.diag(inverted_S_cutoff) @ U.T @ y_train
            y_train_pred_cutoff = X_train @ beta_hat_cutoff
            train_mse_cutoff = mean_squared_error(y_train, y_train_pred_cutoff)
            y_test_pred_cutoff = X_test @ beta_hat_cutoff
            test_mse_cutoff = mean_squared_error(y_test, y_test_pred_cutoff)
            
            dataset_loss_no_small_singular_values_df.append({
                "Dataset": dataset_name,
                "Subset Size": subset_size,
                "Num Parameters": X.shape[1],
                "Train MSE": train_mse_cutoff,
                "Test MSE": test_mse_cutoff,
                "Repeat Index": repeat_idx,
                "Singular Value\nCutoff": cutoff
            })
        
        # Ablation 2: No residuals in ideal fit
        y_train_no_residuals = X_train @ beta_ideal
        y_test_no_residuals = X_test @ beta_ideal
        
        beta_hat_no_residuals = Vt.T @ np.diag(S_inverted) @ U.T @ y_train_no_residuals
        y_train_pred_no_residuals = X_train @ beta_hat_no_residuals
        train_mse_no_residuals = mean_squared_error(y_train_no_residuals, y_train_pred_no_residuals)
        y_test_pred_no_residuals = X_test @ beta_hat_no_residuals
        test_mse_no_residuals = mean_squared_error(y_test_no_residuals, y_test_pred_no_residuals)
        
        dataset_loss_no_residuals_in_ideal_fit_df.append({
            "Dataset": dataset_name,
            "Subset Size": subset_size,
            "Num Parameters": X.shape[1],
            "Train MSE": train_mse_no_residuals,
            "Test MSE": test_mse_no_residuals,
            "Repeat Index": repeat_idx
        })
        
        # Ablation 3: Project test datum features to training feature subspace
        # For Student-Teacher dataset, use these specific values
        num_leading_singular_modes_to_keep = [5, 10, 15, 20, 25]
        
        for num_leading_sing_modes in num_leading_singular_modes_to_keep:
            # Create projection using leading singular modes
            X_train_leading = (
                U[:, :num_leading_sing_modes]
                @ np.diag(S[:num_leading_sing_modes])
                @ Vt[:num_leading_sing_modes, :]
            )
            X_train_pinv_leading = np.linalg.pinv(X_train_leading)
            projection_matrix = np.matmul(X_train_leading.T, X_train_pinv_leading.T)
            
            # Project test features
            X_test_projected = X_test @ projection_matrix.T
            
            # Verify projection is valid
            fraction_inside = np.linalg.norm(X_test_projected, axis=1) / np.linalg.norm(X_test, axis=1)
            assert np.all(np.logical_and(fraction_inside >= -0.001, fraction_inside <= 1.001))
            
            # Make predictions with projected features
            y_test_pred_projected = X_test_projected @ beta_hat_unablated
            test_mse_projected = mean_squared_error(y_test, y_test_pred_projected)
            
            dataset_loss_test_features_in_training_feature_subspace_df.append({
                "Dataset": dataset_name,
                "Subset Size": subset_size,
                "Num Parameters": X.shape[1],
                "Train MSE": train_mse_unablated,  # Same as unablated
                "Test MSE": test_mse_projected,
                "Repeat Index": repeat_idx,
                "Num. Leading\nSingular Modes\nto Keep": num_leading_sing_modes
            })

# Convert to DataFrames and add ratio column
dataset_loss_unablated_df = pd.DataFrame(dataset_loss_unablated_df)
dataset_loss_unablated_df["Num Parameters / Num. Training Samples"] = (
    dataset_loss_unablated_df["Num Parameters"] / dataset_loss_unablated_df["Subset Size"]
)

dataset_loss_no_small_singular_values_df = pd.DataFrame(dataset_loss_no_small_singular_values_df)
dataset_loss_no_small_singular_values_df["Num Parameters / Num. Training Samples"] = (
    dataset_loss_no_small_singular_values_df["Num Parameters"] / 
    dataset_loss_no_small_singular_values_df["Subset Size"]
)

dataset_loss_no_residuals_in_ideal_fit_df = pd.DataFrame(dataset_loss_no_residuals_in_ideal_fit_df)
dataset_loss_no_residuals_in_ideal_fit_df["Num Parameters / Num. Training Samples"] = (
    dataset_loss_no_residuals_in_ideal_fit_df["Num Parameters"] / 
    dataset_loss_no_residuals_in_ideal_fit_df["Subset Size"]
)

dataset_loss_test_features_in_training_feature_subspace_df = pd.DataFrame(
    dataset_loss_test_features_in_training_feature_subspace_df
)
dataset_loss_test_features_in_training_feature_subspace_df["Num Parameters / Num. Training Samples"] = (
    dataset_loss_test_features_in_training_feature_subspace_df["Num Parameters"] / 
    dataset_loss_test_features_in_training_feature_subspace_df["Subset Size"]
)

# Set consistent y limits for plots
ymax = 2 * max(
    dataset_loss_unablated_df.groupby("Subset Size")["Test MSE"].mean().max(),
    dataset_loss_unablated_df.groupby("Subset Size")["Train MSE"].mean().max()
)
ymin = 0.5 * dataset_loss_unablated_df.groupby("Subset Size")["Train MSE"].mean()[X.shape[1] + 1]

# ===== PLOTTING =====

# Plot 1: Unablated MSE
plt.close()
fig, ax = plt.subplots(figsize=(7, 5))
sns.lineplot(
    data=dataset_loss_unablated_df,
    x="Num Parameters / Num. Training Samples",
    y="Train MSE",
    label="Train",
    ax=ax
)
sns.lineplot(
    data=dataset_loss_unablated_df,
    x="Num Parameters / Num. Training Samples",
    y="Test MSE",
    label="Test",
    ax=ax
)
ax.set_xlabel("Num Parameters / Num. Training Samples")
ax.set_ylabel("Mean Squared Error")
ax.axvline(x=1.0, color="black", linestyle="--", label="Interpolation\nThreshold")
ax.set_title(f"Dataset: {dataset_name}")
ax.set_ylim(bottom=ymin, top=ymax)
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
sns.move_legend(obj=ax, loc="upper left", bbox_to_anchor=(1.0, 1.0))
save_plot_with_multiple_extensions(plot_dir=results_dir, plot_title="unablated")

# Plot 2: Smallest Non-Zero Singular Value
plt.close()
fig, ax = plt.subplots(figsize=(6, 5))
sns.lineplot(
    data=dataset_loss_unablated_df,
    x="Num Parameters / Num. Training Samples",
    y="Smallest Non-Zero Singular Value",
    color="green",
    ax=ax
)
ax.set_xlabel("Num Parameters / Num. Training Samples")
ax.set_ylabel("Smallest Non-Zero Singular\nValue of Training Features")
ax.axvline(x=1.0, color="black", linestyle="--")
ax.set_title(f"Dataset: {dataset_name}")
ax.set_xscale("log")
ax.set_yscale("log")
save_plot_with_multiple_extensions(plot_dir=results_dir, plot_title="least_informative_singular_value")

# Plot 3: Test Bias Squared
test_bias_squared_ymin = 0.2 * dataset_loss_unablated_df[
    dataset_loss_unablated_df["Subset Size"] == (X.shape[1] - 1)
]["Test Bias Squared"].mean()

plt.close()
fig, ax = plt.subplots(figsize=(6, 5))
sns.lineplot(
    data=dataset_loss_unablated_df,
    x="Num Parameters / Num. Training Samples",
    y="Test Bias Squared",
    color="purple",
    ax=ax
)
ax.set_xlabel("Num Parameters / Num. Training Samples")
ax.set_ylabel("Test Bias Squared")
ax.axvline(x=1.0, color="black", linestyle="--")
ax.plot(
    [dataset_loss_unablated_df["Num Parameters / Num. Training Samples"].min(), 1.0],
    [test_bias_squared_ymin, test_bias_squared_ymin],
    color="purple",
    linestyle="--",
    label="Test = 0"
)
ax.set_ylim(bottom=test_bias_squared_ymin)
ax.set_title(f"Dataset: {dataset_name}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
save_plot_with_multiple_extensions(plot_dir=results_dir, plot_title="test_bias_squared")

# Plot 4: No Small Singular Values
plt.close()
fig, ax = plt.subplots(figsize=(7, 5))
sns.lineplot(
    data=dataset_loss_no_small_singular_values_df,
    x="Num Parameters / Num. Training Samples",
    y="Train MSE",
    hue="Singular Value\nCutoff",
    legend=False,
    ax=ax,
    hue_norm=LogNorm(),
    palette="PuBu"
)
sns.lineplot(
    data=dataset_loss_no_small_singular_values_df,
    x="Num Parameters / Num. Training Samples",
    y="Test MSE",
    hue="Singular Value\nCutoff",
    ax=ax,
    hue_norm=LogNorm(),
    palette="OrRd"
)
ax.set_xlabel("Num Parameters / Num. Training Samples")
ax.set_title(f"Dataset: {dataset_name}\nAblation: No Small Singular Values")
ax.axvline(x=1.0, color="black", linestyle="--")
ax.set_ylim(bottom=ymin, top=ymax)
ax.set_xscale("log")
ax.set_yscale("log")
sns.move_legend(obj=ax, loc="upper left", bbox_to_anchor=(1.0, 1.0))
save_plot_with_multiple_extensions(plot_dir=results_dir, plot_title="no_small_singular_values")

# Plot 5: Test Features in Training Feature Subspace
plt.close()
fig, ax = plt.subplots(figsize=(7, 5))
sns.lineplot(
    data=dataset_loss_test_features_in_training_feature_subspace_df,
    x="Num Parameters / Num. Training Samples",
    y="Train MSE",
    hue="Num. Leading\nSingular Modes\nto Keep",
    legend=False,
    ax=ax,
    palette="PuBu"
)
sns.lineplot(
    data=dataset_loss_test_features_in_training_feature_subspace_df,
    x="Num Parameters / Num. Training Samples",
    y="Test MSE",
    hue="Num. Leading\nSingular Modes\nto Keep",
    ax=ax,
    palette="OrRd"
)
ax.set_xlabel("Num Parameters / Num. Training Samples")
ax.set_title(f"Dataset: {dataset_name}\nAblation: Test Features in Training Feature Subspace")
ax.axvline(x=1.0, color="black", linestyle="--")
ax.set_ylim(bottom=ymin, top=ymax)
ax.set_xscale("log")
ax.set_yscale("log")
sns.move_legend(obj=ax, loc="upper left", bbox_to_anchor=(1.0, 1.0))
save_plot_with_multiple_extensions(plot_dir=results_dir, plot_title="test_feat_in_train_feat_subspace")

# Plot 6: No Residuals in Ideal Fit
plt.close()
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(
    [dataset_loss_no_residuals_in_ideal_fit_df["Num Parameters / Num. Training Samples"].min(), 1.0],
    [1.1 * ymin, 1.1 * ymin],
    color="tab:blue",
    label="Train = 0"
)
sns.lineplot(
    data=dataset_loss_no_residuals_in_ideal_fit_df,
    x="Num Parameters / Num. Training Samples",
    y="Test MSE",
    label=r"Test $\neq$ 0",
    ax=ax,
    color="tab:orange"
)
ax.plot(
    [dataset_loss_no_residuals_in_ideal_fit_df["Num Parameters / Num. Training Samples"].min(), 1.0],
    [1.1 * ymin, 1.1 * ymin],
    color="tab:orange",
    linestyle="--",
    label="Test = 0"
)
ax.set_xlabel("Num Parameters / Num. Training Samples")
ax.set_title(f"Dataset: {dataset_name}\nAblation: No Residuals in Ideal Fit")
ax.axvline(x=1.0, color="black", linestyle="--")
ax.set_ylim(bottom=ymin, top=ymax)
ax.set_xscale("log")
ax.set_yscale("log")
sns.move_legend(obj=ax, loc="upper left", bbox_to_anchor=(1.0, 1.0))
save_plot_with_multiple_extensions(plot_dir=results_dir, plot_title="no_residuals_in_ideal")

print(f"Analysis complete! Results saved to: {results_dir}")
