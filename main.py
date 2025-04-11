import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os

# Set up reproducibility
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

def compute_gradient_descent_solution(X, y, learning_rate=0.01, n_iterations=1000, tol=1e-6):
    """Compute the gradient descent solution for linear regression."""
    n_samples, n_features = X.shape
    
    # Reshape y to be a column vector if it isn't already
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    # Initialize parameters
    beta = np.zeros((n_features, 1))
    
    # Normalize X for better conditioning
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8  # Add small constant to avoid division by zero
    X_normalized = (X - X_mean) / X_std
    
    # Normalize y
    y_mean = np.mean(y)
    y_std = np.std(y) + 1e-8
    y_normalized = (y - y_mean) / y_std
    
    # Compute adaptive learning rate based on largest eigenvalue
    XtX = X_normalized.T @ X_normalized
    eigenvalues = np.linalg.eigvalsh(XtX)
    max_eigenvalue = np.max(np.abs(eigenvalues)) + 1e-8
    learning_rate = min(learning_rate, 1.0 / max_eigenvalue)
    
    # Initialize best solution tracking
    best_beta = beta.copy()
    best_loss = float('inf')
    
    for _ in range(n_iterations):
        # Compute predictions
        y_pred = X_normalized @ beta
        
        # Compute loss
        current_loss = np.mean((y_pred - y_normalized) ** 2)
        
        # Update best solution if current is better
        if current_loss < best_loss:
            best_loss = current_loss
            best_beta = beta.copy()
        
        # Compute gradient with normalization
        gradient = (2/n_samples) * X_normalized.T @ (y_pred - y_normalized)
        
        # Update parameters with gradient clipping
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > 1.0:
            gradient = gradient / gradient_norm
            
        beta_new = beta - learning_rate * gradient
        
        # Check convergence
        if np.all(np.abs(beta_new - beta) < tol):
            break
            
        beta = beta_new
    
    # Use the best solution found
    beta = best_beta
    
    # Denormalize the coefficients
    beta = (y_std / X_std.reshape(-1, 1)) * beta
    beta = beta - np.sum(X_mean.reshape(-1, 1) * beta) + y_mean
    
    return beta

def save_plot_with_multiple_extensions(plot_dir, plot_title):
    """Save the current plot in multiple file formats."""
    for extension in ['png', 'pdf', 'svg']:
        plt.savefig(
            os.path.join(plot_dir, f"{plot_title}.{extension}"),
            bbox_inches="tight",
            dpi=300
        )

def analyze_synthetic_data():
    """Run analysis on synthetic data and generate plots."""
    # Create results directory
    results_dir = "results/synthetic_data_analysis"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get synthetic data
    X, y = generate_synthetic_data(return_X_y=True)
    
    # Compute ideal coefficients for linear relationship
    beta_ideal = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # Initialize data collection
    dataset_loss_unablated_df = []
    dataset_loss_no_small_singular_values_df = []
    dataset_loss_no_residuals_in_ideal_fit_df = []
    dataset_loss_test_features_in_training_feature_subspace_df = []
    
    # Set parameters for analysis
    num_repeats = 10  # Reduced from 50 for efficiency
    singular_value_cutoffs = np.logspace(-3, 0, 7)
    
    for repeat_idx in range(num_repeats):
        subset_sizes = np.arange(1, 40, 2)  # Step by 2 for efficiency
        for subset_size in subset_sizes:
            print(f"Repeat: {repeat_idx}, Subset size: {subset_size}")
            
            # Split the data
            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
                X, y, np.arange(X.shape[0]), 
                random_state=repeat_idx,
                test_size=X.shape[0] - subset_size,
                shuffle=True
            )
            
            # Unablated linear fit
            beta_hat_unablated = compute_gradient_descent_solution(X_train, y_train)
            y_train_pred = X_train @ beta_hat_unablated
            train_mse_unablated = mean_squared_error(y_train, y_train_pred)
            y_test_pred = X_test @ beta_hat_unablated
            test_mse_unablated = mean_squared_error(y_test, y_test_pred)
            
            # SVD analysis
            U, S, Vt = np.linalg.svd(X_train, full_matrices=False, compute_uv=True)
            min_singular_value = np.min(S[S > 0.0])
            
            # Calculate projection for test bias
            X_hat_test = X_test @ X_train.T @ np.linalg.pinv(X_train @ X_train.T) @ X_train
            X_test_diff = X_hat_test - X_test
            X_test_diff_inner_beta_ideal = np.mean(X_test_diff @ beta_ideal)
            
            # Store unablated results
            dataset_loss_unablated_df.append({
                "Subset Size": subset_size,
                "Num Parameters": X.shape[1],
                "Train MSE": train_mse_unablated,
                "Test MSE": test_mse_unablated,
                "Repeat Index": repeat_idx,
                "Test Bias Squared": np.square(X_test_diff_inner_beta_ideal),
                "Smallest Non-Zero Singular Value": min_singular_value
            })
            
            # Ablation: No small singular values
            for cutoff in singular_value_cutoffs:
                S_mask = S >= cutoff
                if np.any(S_mask):
                    X_train_filtered = (U[:, S_mask] @ np.diag(S[S_mask]) @ Vt[S_mask, :])
                    X_test_filtered = X_test @ Vt[S_mask, :].T @ Vt[S_mask, :]
                    
                    beta_hat_cutoff = compute_gradient_descent_solution(X_train_filtered, y_train)
                    y_train_pred_cutoff = X_train_filtered @ beta_hat_cutoff
                    y_test_pred_cutoff = X_test_filtered @ beta_hat_cutoff
                else:
                    y_train_pred_cutoff = np.zeros_like(y_train)
                    y_test_pred_cutoff = np.zeros_like(y_test)
                
                train_mse_cutoff = mean_squared_error(y_train, y_train_pred_cutoff)
                test_mse_cutoff = mean_squared_error(y_test, y_test_pred_cutoff)
                
                dataset_loss_no_small_singular_values_df.append({
                    "Subset Size": subset_size,
                    "Num Parameters": X.shape[1],
                    "Train MSE": train_mse_cutoff,
                    "Test MSE": test_mse_cutoff,
                    "Repeat Index": repeat_idx,
                    "Singular Value Cutoff": cutoff
                })
            
            # Ablation: No residuals in ideal fit
            y_train_no_residuals = X_train @ beta_ideal
            y_test_no_residuals = X_test @ beta_ideal
            
            beta_hat_no_residuals = compute_gradient_descent_solution(X_train, y_train_no_residuals)
            y_train_pred_no_residuals = X_train @ beta_hat_no_residuals
            y_test_pred_no_residuals = X_test @ beta_hat_no_residuals
            
            train_mse_no_residuals = mean_squared_error(y_train_no_residuals, y_train_pred_no_residuals)
            test_mse_no_residuals = mean_squared_error(y_test_no_residuals, y_test_pred_no_residuals)
            
            dataset_loss_no_residuals_in_ideal_fit_df.append({
                "Subset Size": subset_size,
                "Num Parameters": X.shape[1],
                "Train MSE": train_mse_no_residuals,
                "Test MSE": test_mse_no_residuals,
                "Repeat Index": repeat_idx
            })
            
            # Ablation: Project test features to training feature subspace
            train_mse_test_features_in_training_feature_subspace = train_mse_unablated
            num_leading_singular_modes_to_keep = [5, 10, 15, 20, 25]  # For Student-Teacher dataset
            
            for num_leading_sing_modes in num_leading_singular_modes_to_keep:
                X_train_leading = (
                    U[:, :num_leading_sing_modes]
                    @ np.diag(S[:num_leading_sing_modes])
                    @ Vt[:num_leading_sing_modes, :]
                )
                X_train_pinv_leading = np.linalg.pinv(X_train_leading)
                projection_matrix = np.matmul(X_train_leading.T, X_train_pinv_leading.T)
                X_test_projected_onto_leading_X_train_modes = X_test @ projection_matrix.T
                
                fraction_inside = np.linalg.norm(X_test_projected_onto_leading_X_train_modes, axis=1) / np.linalg.norm(X_test, axis=1)
                
                y_test_pred_projected_onto_leading_train_modes = X_test_projected_onto_leading_X_train_modes @ beta_hat_unablated
                test_mse_test_features_in_training_feature_subspace = mean_squared_error(
                    y_test, y_test_pred_projected_onto_leading_train_modes
                )
                
                dataset_loss_test_features_in_training_feature_subspace_df.append({
                    "Subset Size": subset_size,
                    "Num Parameters": X.shape[1],
                    "Train MSE": train_mse_test_features_in_training_feature_subspace,
                    "Test MSE": test_mse_test_features_in_training_feature_subspace,
                    "Repeat Index": repeat_idx,
                    "Num Leading Singular Modes": num_leading_sing_modes
                })
    
    # Convert to DataFrames and add ratio column
    dataset_loss_unablated_df = pd.DataFrame(dataset_loss_unablated_df)
    dataset_loss_unablated_df["Num Parameters / Num Training Samples"] = (
        dataset_loss_unablated_df["Num Parameters"] / dataset_loss_unablated_df["Subset Size"]
    )
    
    dataset_loss_no_small_singular_values_df = pd.DataFrame(dataset_loss_no_small_singular_values_df)
    dataset_loss_no_small_singular_values_df["Num Parameters / Num Training Samples"] = (
        dataset_loss_no_small_singular_values_df["Num Parameters"] / 
        dataset_loss_no_small_singular_values_df["Subset Size"]
    )
    
    dataset_loss_no_residuals_in_ideal_fit_df = pd.DataFrame(dataset_loss_no_residuals_in_ideal_fit_df)
    dataset_loss_no_residuals_in_ideal_fit_df["Num Parameters / Num Training Samples"] = (
        dataset_loss_no_residuals_in_ideal_fit_df["Num Parameters"] / 
        dataset_loss_no_residuals_in_ideal_fit_df["Subset Size"]
    )
    
    dataset_loss_test_features_in_training_feature_subspace_df = pd.DataFrame(
        dataset_loss_test_features_in_training_feature_subspace_df
    )
    dataset_loss_test_features_in_training_feature_subspace_df["Num Parameters / Num Training Samples"] = (
        dataset_loss_test_features_in_training_feature_subspace_df["Num Parameters"] / 
        dataset_loss_test_features_in_training_feature_subspace_df["Subset Size"]
    )
    
    # Set consistent y limits for plots
    ymax = 2 * max(
        dataset_loss_unablated_df.groupby("Subset Size")["Test MSE"].mean().max(),
        dataset_loss_unablated_df.groupby("Subset Size")["Train MSE"].mean().max()
    )
    ymin = 0.5 * dataset_loss_unablated_df.groupby("Subset Size")["Train MSE"].mean()[X.shape[1] + 1]
    
    # Generate plots
    generate_plots(
        results_dir, 
        dataset_loss_unablated_df, 
        dataset_loss_no_small_singular_values_df,
        dataset_loss_no_residuals_in_ideal_fit_df,
        dataset_loss_test_features_in_training_feature_subspace_df,
        ymin, ymax
    )
    
    return {
        "unablated": dataset_loss_unablated_df,
        "no_small_singular_values": dataset_loss_no_small_singular_values_df,
        "no_residuals": dataset_loss_no_residuals_in_ideal_fit_df,
        "test_features_projection": dataset_loss_test_features_in_training_feature_subspace_df
    }

def generate_plots(results_dir, unablated_df, no_small_sv_df, no_residuals_df, test_proj_df, ymin, ymax):
    """Generate all plots for the analysis."""
    # Plot 1: Unablated MSE
    plt.figure(figsize=(7, 5))
    sns.lineplot(
        data=unablated_df,
        x="Num Parameters / Num Training Samples",
        y="Train MSE",
        label="Train"
    )
    sns.lineplot(
        data=unablated_df,
        x="Num Parameters / Num Training Samples",
        y="Test MSE",
        label="Test"
    )
    plt.xlabel("Num Parameters / Num Training Samples")
    plt.ylabel("Mean Squared Error")
    plt.axvline(x=1.0, color="black", linestyle="--", label="Interpolation\nThreshold")
    plt.title("Synthetic Data: Student-Teacher Model")
    plt.ylim(bottom=ymin, top=ymax)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save_plot_with_multiple_extensions(results_dir, "unablated")
    plt.close()
    
    # Plot 2: Singular Values
    plt.figure(figsize=(6, 5))
    sns.lineplot(
        data=unablated_df,
        x="Num Parameters / Num Training Samples",
        y="Smallest Non-Zero Singular Value",
        color="green"
    )
    plt.xlabel("Num Parameters / Num Training Samples")
    plt.ylabel("Smallest Non-Zero Singular\nValue of Training Features")
    plt.axvline(x=1.0, color="black", linestyle="--")
    plt.title("Synthetic Data: Student-Teacher Model")
    plt.xscale("log")
    plt.yscale("log")
    save_plot_with_multiple_extensions(results_dir, "least_informative_singular_value")
    plt.close()
    
    # Plot 3: Test Bias Squared
    test_bias_squared_ymin = 0.2 * unablated_df[
        unablated_df["Subset Size"] == (unablated_df["Num Parameters"].iloc[0] - 1)
    ]["Test Bias Squared"].mean()
    
    plt.figure(figsize=(6, 5))
    sns.lineplot(
        data=unablated_df,
        x="Num Parameters / Num Training Samples",
        y="Test Bias Squared",
        color="purple"
    )
    plt.xlabel("Num Parameters / Num Training Samples")
    plt.ylabel("Test Bias Squared")
    plt.axvline(x=1.0, color="black", linestyle="--")
    plt.plot(
        [unablated_df["Num Parameters / Num Training Samples"].min(), 1.0],
        [test_bias_squared_ymin, test_bias_squared_ymin],
        color="purple",
        linestyle="--",
        label="Test = 0"
    )
    plt.ylim(bottom=test_bias_squared_ymin)
    plt.title("Synthetic Data: Student-Teacher Model")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    save_plot_with_multiple_extensions(results_dir, "test_bias_squared")
    plt.close()
    
    # Plot 4: No Small Singular Values
    plt.figure(figsize=(7, 5))
    from matplotlib.colors import LogNorm
    
    sns.lineplot(
        data=no_small_sv_df,
        x="Num Parameters / Num Training Samples",
        y="Train MSE",
        hue="Singular Value Cutoff",
        legend=False,
        hue_norm=LogNorm(),
        palette="PuBu"
    )
    sns.lineplot(
        data=no_small_sv_df,
        x="Num Parameters / Num Training Samples",
        y="Test MSE",
        hue="Singular Value Cutoff",
        hue_norm=LogNorm(),
        palette="OrRd"
    )
    plt.xlabel("Num Parameters / Num Training Samples")
    plt.title("Synthetic Data: Student-Teacher Model\nAblation: No Small Singular Values")
    plt.axvline(x=1.0, color="black", linestyle="--")
    plt.ylim(bottom=ymin, top=ymax)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save_plot_with_multiple_extensions(results_dir, "no_small_singular_values")
    plt.close()
    
    # Plot 5: Test Features in Training Feature Subspace
    plt.figure(figsize=(7, 5))
    sns.lineplot(
        data=test_proj_df,
        x="Num Parameters / Num Training Samples",
        y="Train MSE",
        hue="Num Leading Singular Modes",
        legend=False,
        palette="PuBu"
    )
    sns.lineplot(
        data=test_proj_df,
        x="Num Parameters / Num Training Samples",
        y="Test MSE",
        hue="Num Leading Singular Modes",
        palette="OrRd"
    )
    plt.xlabel("Num Parameters / Num Training Samples")
    plt.title("Synthetic Data: Student-Teacher Model\nAblation: Test Features in Training Feature Subspace")
    plt.axvline(x=1.0, color="black", linestyle="--")
    plt.ylim(bottom=ymin, top=ymax)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save_plot_with_multiple_extensions(results_dir, "test_feat_in_train_feat_subspace")
    plt.close()
    
    # Plot 6: No Residuals in Ideal Fit
    plt.figure(figsize=(7, 5))
    plt.plot(
        [no_residuals_df["Num Parameters / Num Training Samples"].min(), 1.0],
        [1.1 * ymin, 1.1 * ymin],
        color="tab:blue",
        label="Train = 0"
    )
    sns.lineplot(
        data=no_residuals_df,
        x="Num Parameters / Num Training Samples",
        y="Test MSE",
        label="Test ≠ 0",
        color="tab:orange"
    )
    plt.plot(
        [no_residuals_df["Num Parameters / Num Training Samples"].min(), 1.0],
        [1.1 * ymin, 1.1 * ymin],
        color="tab:orange",
        linestyle="--",
        label="Test = 0"
    )
    plt.xlabel("Num Parameters / Num Training Samples")
    plt.title("Synthetic Data: Student-Teacher Model\nAblation: No Residuals in Ideal Fit")
    plt.axvline(x=1.0, color="black", linestyle="--")
    plt.ylim(bottom=ymin, top=ymax)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save_plot_with_multiple_extensions(results_dir, "no_residuals_in_ideal")
    plt.close()

if __name__ == "__main__":
    results = analyze_synthetic_data()
    print("Analysis complete. Results saved to 'results/synthetic_data_analysis' directory.")
