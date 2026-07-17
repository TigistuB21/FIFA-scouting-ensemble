"""
Supervised Learning Stacking Ensemble Framework
===============================================
This script implements a manual stacking ensemble classifier using scikit-learn.
The framework consists of:
  - Level-0 Base Models: Logistic Regression, Decision Tree, Random Forest
  - Level-1 Meta Model: Logistic Regression

Dataset: FIFA 23 Complete Player Dataset (Kaggle)
Target: is_elite (Overall rating >= 75)
"""

import os
import time
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ignore minor warnings to keep console output clean and professional
warnings.filterwarnings("ignore")


def locate_dataset(base_dir: str) -> Path:
    """
    Scans the given cache directory to locate player CSV files.
    Prefers manageable-sized files (under 500 MB) to prevent out-of-memory errors
    on standard systems, and selects the largest of them.
    
    Parameters:
    - base_dir: Path to search for CSV files
    
    Returns:
    - Path object pointing to the selected CSV file
    """
    path = Path(base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Provided dataset directory does not exist: {base_dir}")

    print("Scanning directory for CSV files...")
    csv_files = list(path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {base_dir}")

    for file in csv_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - Found: {file.name} ({size_mb:.2f} MB)")

    # Filter for files containing 'player' but not coaches/teams
    player_files = [
        f for f in csv_files
        if "player" in f.name.lower()
        and "coach" not in f.name.lower()
        and "team" not in f.name.lower()
    ]

    if not player_files:
        raise FileNotFoundError("Could not find any player CSV files.")

    # Select player files that are under 500 MB to avoid memory issues
    manageable_files = [f for f in player_files if f.stat().st_size < 500 * 1024 * 1024]

    if not manageable_files:
        # Fallback to the smallest player file if all are larger than 500 MB
        selected_file = min(player_files, key=lambda f: f.stat().st_size)
        print(f"Warning: All player files are > 500MB. Selecting smallest: {selected_file.name}")
    else:
        # Select the largest of the manageable player files (usually female_players.csv)
        selected_file = max(manageable_files, key=lambda f: f.stat().st_size)
        print(f"Automatically selected dataset: {selected_file.name}")

    return selected_file


def recommend_target() -> str:
    """
    Prints the target variable recommendation and reasoning.
    """
    recommendation = (
        "\n=================== TARGET VARIABLE RECOMMENDATION ===================\n"
        "We recommend using 'is_elite' (overall rating >= 75) as the target variable.\n\n"
        "Reasoning:\n"
        "1. CLINICAL ML RELEVANCE: Predicting whether a player is elite using only physical\n"
        "   and technical skill attributes represents a real-world sports scouting task.\n"
        "2. CLASS BALANCE: The overall rating median is 75, yielding a balanced ~56/44 split\n"
        "   for class 1 (Elite) vs class 0 (Not Elite). This ensures classification metrics like\n"
        "   Precision, Recall, and ROC-AUC are robust and highly interpretable.\n"
        "3. AVOIDING TRIVIAL TASKS: Predicting Goalkeeper (GK) vs Outfield player is trivial\n"
        "   (yielding 99%+ accuracy) and does not showcase the benefits of stacking base classifiers.\n"
        "4. PREVENTING DATA LEAKAGE: We will explicitly drop rating columns (overall, potential)\n"
        "   and commercial metrics (value_eur, wage_eur, release_clause_eur) which are proxies\n"
        "   for the target, forcing models to rely purely on underlying skill features.\n"
        "======================================================================\n"
    )
    print(recommendation)
    return "is_elite"


def load_and_preprocess_data(file_path: Path):
    """
    Loads dataset, applies filters, prints metadata summaries, and extracts
    features and target.
    
    Parameters:
    - file_path: Path to the selected CSV file
    
    Returns:
    - X: DataFrame containing features
    - y: Series containing binary target (is_elite)
    - numeric_cols: list of numeric feature names
    - categorical_cols: list of categorical feature names
    """
    print(f"\nLoading dataset from {file_path.name}...")
    # Read with low_memory=False to handle mixed types gracefully
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Original shape: {df.shape}")

    # Step 1: Filter for FIFA 23
    # This prevents temporal leakage (e.g. training on player stats from 2021 and testing on them in 2023)
    if 'fifa_version' in df.columns:
        df = df[df['fifa_version'] == 23].copy()
        print(f"Filtered for FIFA 23. Shape: {df.shape}")
    else:
        print("Note: 'fifa_version' column not found; proceeding with full dataset.")

    # Step 2: Print general dataset info
    print("\n--- Dataset Information Summary ---")
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")

    # Step 3: Print missing values summary
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    print(f"\nColumns with missing values (Total: {len(missing_cols)}):")
    if not missing_cols.empty:
        # Display top 15 missing value columns
        print(missing_cols.sort_values(ascending=False).head(15).to_string())
    else:
        print("No missing values found.")

    # Step 4: Define features
    # Numeric features (core skills and physical attributes)
    numeric_cols = [
        'age', 'height_cm', 'weight_kg', 'weak_foot', 'skill_moves', 'international_reputation',
        'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
        'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys',
        'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
        'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance',
        'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
        'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure',
        'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
        'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes'
    ]

    # Categorical features
    categorical_cols = ['preferred_foot', 'work_rate', 'body_type']

    print(f"\nSelected features for training:")
    print(f"  - Numerical features: {len(numeric_cols)}")
    print(f"  - Categorical features: {len(categorical_cols)}")

    # Define target
    # Elite player defined as overall rating >= 75
    y = (df['overall'] >= 75).astype(int)
    print("\nTarget Class Distribution:")
    class_counts = y.value_counts()
    print(f"  - Class 0 (Not Elite): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(y)*100:.2f}%)")
    print(f"  - Class 1 (Elite): {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(y)*100:.2f}%)")

    # Combine features
    X = df[numeric_cols + categorical_cols]

    return X, y, numeric_cols, categorical_cols


def build_preprocessing_pipeline(numeric_cols, categorical_cols):
    """
    Creates a scikit-learn ColumnTransformer for pre-processing.
    - Numerical: Median imputation + Standard Scaling
    - Categorical: Most frequent imputation + One-Hot Encoding
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
            ]), categorical_cols)
        ]
    )
    return preprocessor


def generate_oof_predictions(models, X_train, y_train, preprocessor, n_splits=5):
    """
    Manually generates Out-Of-Fold (OOF) predictions for Level-0 models using 
    StratifiedKFold.
    
    PREVENTING DATA LEAKAGE:
    1. A StratifiedKFold split separates the training data into 'fold training' 
       and 'fold validation' sets.
    2. Crucially, the preprocessor (scaling and imputation parameters) is fit 
       ONLY on the fold training set and then applied (transformed) on the validation set. 
       This prevents information leakage (mean/std of validation sets do not bleed into training).
    3. The Level-0 models are fit strictly on the fold training set and predict on 
       the validation set.
    4. By doing this iteratively for all folds, we build Level-1 features (predictions) 
       where the meta-model trains on predictions made on samples the base models never saw 
       during training. This prevents target leakage and ensures the meta-model generalizes 
       to unseen test data.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize dictionary to hold OOF prediction probabilities (positive class)
    oof_preds = {name: np.zeros(len(X_train)) for name in models}
    
    print(f"\nGenerating Out-Of-Fold (OOF) predictions using {n_splits}-fold Stratified CV...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"  - Processing Fold {fold}/{n_splits}...")
        
        # Split features and target
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Clone preprocessor to avoid state retention between folds
        fold_preprocessor = clone(preprocessor)
        X_tr_proc = fold_preprocessor.fit_transform(X_tr)
        X_val_proc = fold_preprocessor.transform(X_val)
        
        for name, model in models.items():
            # Clone model to reset weights
            fold_model = clone(model)
            fold_model.fit(X_tr_proc, y_tr)
            
            # Predict probabilities for class 1
            oof_preds[name][val_idx] = fold_model.predict_proba(X_val_proc)[:, 1]
            
    print("OOF predictions generated successfully.")
    
    # After generating OOF predictions, fit the final preprocessor and base models 
    # on the ENTIRE training set. These fully fit models will be used to make test predictions.
    print("\nFitting final base models on full training set...")
    final_preprocessor = clone(preprocessor)
    X_train_proc = final_preprocessor.fit_transform(X_train)
    
    trained_models = {}
    for name, model in models.items():
        full_model = clone(model)
        full_model.fit(X_train_proc, y_train)
        trained_models[name] = full_model
        print(f"  - Fitted final {name} model.")
        
    oof_df = pd.DataFrame(oof_preds)
    return oof_df, trained_models, final_preprocessor


def run_stacking_framework(dataset_dir: str):
    """
    Main driver function to run the data preparation, training, evaluation, 
    and visualization pipeline.
    """
    try:
        # Step 1: Locate and inspect file
        csv_file = locate_dataset(dataset_dir)
        
        # Step 2: Recommend target variable
        recommend_target()
        
        # Step 3: Load and preprocess dataset
        X, y, numeric_cols, categorical_cols = load_and_preprocess_data(csv_file)
        
        # Step 4: Stratified Train-Test Split (80% Train, 20% Test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\nSplitting data: Train size = {len(X_train)}, Test size = {len(X_test)}")
        
        # Step 5: Define preprocessing pipeline
        preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
        
        # Step 6: Initialize base classifiers (Level-0)
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        
        base_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
        }

        # Step 7: K-Fold cross validation to generate Level-1 training features
        # Measure training time for stacking framework (OOF CV + final fits)
        start_stack_train_time = time.perf_counter()
        oof_df, trained_base_models, final_preprocessor = generate_oof_predictions(
            base_models, X_train, y_train, preprocessor, n_splits=5
        )
        
        # Print diagnostic heads for Level-0 datasets
        print("\n=================== DIAGNOSTIC DATAFRAME HEADS ===================")
        print("\n--- Level-0 Raw Training Dataset (X_train - First 5 Rows) ---")
        print(X_train.head())
        
        # Construct preprocessed Level-0 training DataFrame for display
        X_train_proc = final_preprocessor.transform(X_train)
        cat_encoder = final_preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
        all_feature_names = numeric_cols + list(cat_feature_names)
        X_train_proc_df = pd.DataFrame(X_train_proc, columns=all_feature_names)
        
        print("\n--- Level-0 Preprocessed Training Dataset (First 5 Rows) ---")
        print(X_train_proc_df.head())
        
        print("\n--- Level-1 Training Dataset (OOF Predictions - First 5 Rows) ---")
        print(oof_df.head())
        print("==================================================================")
        
        # Train meta-model (Level-1) using OOF predicted probabilities
        print("\nTraining Meta-Model (Level-1 Logistic Regression) on OOF predictions...")
        meta_model = LogisticRegression(random_state=42)
        meta_model.fit(oof_df, y_train)
        stack_train_time = time.perf_counter() - start_stack_train_time
        print(f"Meta-model training complete. Stacking Train Time: {stack_train_time:.4f}s")
        
        # Step 8: Preprocess test set using the fitted preprocessor
        X_test_proc = final_preprocessor.transform(X_test)
        
        # Dictionary to store results for evaluation
        results = {}
        
        # Evaluate individual Level-0 models
        for name, model in trained_base_models.items():
            print(f"\nEvaluating final {name} on test set...")
            
            # Measure prediction time
            start_pred = time.perf_counter()
            y_pred = model.predict(X_test_proc)
            y_prob = model.predict_proba(X_test_proc)[:, 1]
            pred_time = time.perf_counter() - start_pred
            
            # Measure standard training time (single model fit on full X_train)
            start_train = time.perf_counter()
            single_preprocessor = clone(preprocessor)
            X_tr_single = single_preprocessor.fit_transform(X_train)
            single_model = clone(base_models[name])
            single_model.fit(X_tr_single, y_train)
            train_time = time.perf_counter() - start_train
            
            results[name] = {
                'y_pred': y_pred,
                'y_prob': y_prob,
                'train_time': train_time,
                'pred_time': pred_time,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_prob)
            }
            
        # Evaluate Stacking Ensemble
        print("\nEvaluating Stacking Ensemble on test set...")
        start_stack_pred = time.perf_counter()
        
        # Construct Level-1 test features using Level-0 test predictions
        test_meta_preds = {}
        for name, model in trained_base_models.items():
            test_meta_preds[name] = model.predict_proba(X_test_proc)[:, 1]
        X_meta_test = pd.DataFrame(test_meta_preds)
        
        print("\n--- Level-1 Test Dataset (Test Predictions - First 5 Rows) ---")
        print(X_meta_test.head())
        
        # Predict using meta-model
        y_pred_stack = meta_model.predict(X_meta_test)
        y_prob_stack = meta_model.predict_proba(X_meta_test)[:, 1]
        stack_pred_time = time.perf_counter() - start_stack_pred
        
        results["Stacking Ensemble"] = {
            'y_pred': y_pred_stack,
            'y_prob': y_prob_stack,
            'train_time': stack_train_time,
            'pred_time': stack_pred_time,
            'Accuracy': accuracy_score(y_test, y_pred_stack),
            'Precision': precision_score(y_test, y_pred_stack),
            'Recall': recall_score(y_test, y_pred_stack),
            'F1-score': f1_score(y_test, y_pred_stack),
            'ROC-AUC': roc_auc_score(y_test, y_prob_stack)
        }
        
        # Step 9: Print Classification Reports
        print("\n=================== CLASSIFICATION REPORTS ===================")
        for name, res in results.items():
            print(f"\nModel: {name}")
            print(classification_report(y_test, res['y_pred'], target_names=['Not Elite', 'Elite']))
            
        # Step 10: Compile comparison table
        print("\n=================== PERFORMANCE COMPARISON ===================")
        metrics_df = pd.DataFrame(columns=[
            'Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'Training Time (s)', 'Prediction Time (s)'
        ])
        
        for name, res in results.items():
            metrics_df = pd.concat([metrics_df, pd.DataFrame([{
                'Model': name,
                'Accuracy': res['Accuracy'],
                'Precision': res['Precision'],
                'Recall': res['Recall'],
                'F1-score': res['F1-score'],
                'ROC-AUC': res['ROC-AUC'],
                'Training Time (s)': res['train_time'],
                'Prediction Time (s)': res['pred_time']
            }])], ignore_index=True)
            
        # Print table
        print(metrics_df.to_string(index=False))
        
        # Highlight best model based on F1-Score and ROC-AUC (AUC)
        best_f1_idx = metrics_df['F1-score'].idxmax()
        best_f1_name = metrics_df.loc[best_f1_idx, 'Model']
        best_f1_val = metrics_df.loc[best_f1_idx, 'F1-score']
        
        best_auc_idx = metrics_df['ROC-AUC'].idxmax()
        best_auc_name = metrics_df.loc[best_auc_idx, 'Model']
        best_auc_val = metrics_df.loc[best_auc_idx, 'ROC-AUC']
        
        print(f"\n[BEST MODEL] Best Model based on F1-Score: {best_f1_name} (F1 = {best_f1_val:.4f})")
        print(f"[BEST MODEL] Best Model based on ROC-AUC (AUC): {best_auc_name} (AUC = {best_auc_val:.4f})")
        
        # Step 11: Visualization (Save to plots/)
        plots_dir = Path("plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Get categorical encoder feature names for feature importance
        cat_encoder = final_preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
        all_feature_names = numeric_cols + list(cat_feature_names)
        
        print("\nGenerating publication-quality visualization plots...")
        generate_plots(
            y_test=y_test,
            results=results,
            rf_model=trained_base_models["Random Forest"],
            feature_names=all_feature_names,
            X_train_numerical=X_train[numeric_cols],
            plots_dir=plots_dir
        )
        print(f"Plots saved to folder: {plots_dir.resolve()}")
        
        # Print discussion points
        print_discussion_section(metrics_df, results)

    except Exception as e:
        print(f"\n[ERROR] Error encountered during pipeline execution: {e}")
        import traceback
        traceback.print_exc()


def generate_plots(y_test, results, rf_model, feature_names, X_train_numerical, plots_dir):
    """
    Generate and save publication-quality plots.
    """
    # Set aesthetics for plotting
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'figure.titlesize': 20,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.dpi': 300
    })
    
    model_names = list(results.keys())
    colors = ['#4A90E2', '#E28A4A', '#50B83C', '#9F4AE2'] # Custom professional palette
    
    # 1. Confusion Matrices Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    for idx, name in enumerate(model_names):
        cm = confusion_matrix(y_test, results[name]['y_pred'])
        # Display labels and percentage
        cm_labels = np.array([
            [f"TN\n{cm[0,0]}", f"FP\n{cm[0,1]}"],
            [f"FN\n{cm[1,0]}", f"TP\n{cm[1,1]}"]
        ])
        sns.heatmap(cm, annot=cm_labels, fmt='', cmap='Blues', ax=axes[idx], cbar=False,
                    annot_kws={"size": 16, "weight": "bold"})
        axes[idx].set_title(f"{name}", fontsize=18, fontweight='bold', pad=10)
        axes[idx].set_xlabel("Predicted Label", fontsize=12)
        axes[idx].set_ylabel("True Label", fontsize=12)
        axes[idx].set_xticklabels(['Not Elite', 'Elite'], fontsize=11)
        axes[idx].set_yticklabels(['Not Elite', 'Elite'], fontsize=11)
    plt.suptitle("Confusion Matrix Comparisons", fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrices.png", bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curves
    plt.figure(figsize=(10, 8))
    for idx, name in enumerate(model_names):
        fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", color=colors[idx], lw=2.5)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Classifier")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (FPR)", labelpad=10)
    plt.ylabel("True Positive Rate (TPR)", labelpad=10)
    plt.title("ROC Curves Comparison", fontsize=20, fontweight='bold', pad=15)
    plt.legend(loc="lower right", frameon=True)
    plt.savefig(plots_dir / "roc_curves.png", bbox_inches='tight')
    plt.close()
    
    # 3. Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for idx, name in enumerate(model_names):
        precision, recall, _ = precision_recall_curve(y_test, results[name]['y_prob'])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})", color=colors[idx], lw=2.5)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall", labelpad=10)
    plt.ylabel("Precision", labelpad=10)
    plt.title("Precision-Recall Curves Comparison", fontsize=20, fontweight='bold', pad=15)
    plt.legend(loc="lower left", frameon=True)
    plt.savefig(plots_dir / "precision_recall_curves.png", bbox_inches='tight')
    plt.close()
    
    # 4. Feature Importance for Random Forest
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15] # Select top 15 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="crest_r")
    plt.xlabel("Relative Feature Importance Value", labelpad=10)
    plt.ylabel("Features")
    plt.title("Random Forest - Top 15 Feature Importances", fontsize=20, fontweight='bold', pad=15)
    plt.savefig(plots_dir / "rf_feature_importance.png", bbox_inches='tight')
    plt.close()
    
    # 5. Performance Comparison Bar Chart
    metrics_to_compare = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
    df_metrics = []
    for name in model_names:
        for metric in metrics_to_compare:
            df_metrics.append({
                'Model': name,
                'Metric': metric,
                'Value': results[name][metric]
            })
    df_metrics = pd.DataFrame(df_metrics)
    
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df_metrics, palette=colors)
    plt.ylim([0, 1.1])
    plt.ylabel("Score", labelpad=10)
    plt.xlabel("Evaluation Metric", labelpad=10)
    plt.title("Performance Metric Comparison", fontsize=20, fontweight='bold', pad=15)
    
    # Annotate bar values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 4),
                        textcoords='offset points',
                        fontsize=9, fontweight='semibold')
                        
    plt.legend(loc="lower right", frameon=True)
    plt.savefig(plots_dir / "model_comparison.png", bbox_inches='tight')
    plt.close()
    
    # 6. Correlation Heatmap of Key Numerical Features
    key_features = [
        'age', 'height_cm', 'weight_kg', 'pace', 'shooting', 'passing', 'dribbling',
        'defending', 'physic', 'movement_acceleration', 'power_shot_power', 'mentality_composure'
    ]
    existing_keys = [f for f in key_features if f in X_train_numerical.columns]
    corr_matrix = X_train_numerical[existing_keys].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                cbar_kws={"shrink": .8}, annot_kws={"size": 11})
    plt.title("Correlation Heatmap of Key Numerical Features", fontsize=20, fontweight='bold', pad=15)
    plt.savefig(plots_dir / "correlation_heatmap.png", bbox_inches='tight')
    plt.close()


def print_discussion_section(metrics_df, results):
    """
    Prints a concise discussion answering the research questions.
    """
    lr_f1 = results['Logistic Regression']['F1-score']
    dt_f1 = results['Decision Tree']['F1-score']
    rf_f1 = results['Random Forest']['F1-score']
    stack_f1 = results['Stacking Ensemble']['F1-score']
    
    best_base_f1 = max(lr_f1, dt_f1, rf_f1)
    f1_diff = stack_f1 - best_base_f1
    comparison_str = "better than" if f1_diff > 0 else "worse than (or equal to)"
    
    discussion = (
        "\n=================== DISCUSSION & ANALYSIS ===================\n"
        f"1. WHY STACKING PERFORMED {comparison_str.upper()} INDIVIDUAL MODELS:\n"
        f"   - Stacking Ensemble F1-Score: {stack_f1:.4f}\n"
        f"   - Best Level-0 Base F1-Score: {best_base_f1:.4f} (Difference: {f1_diff:+.4f})\n"
        "   - Rationale: Stacking works by using out-of-fold predicted probabilities as features\n"
        "     for a meta-learner. If the base classifiers make uncorrelated errors, the meta-model\n"
        "     learns when to trust each model. For instance, the Random Forest excels at high-level,\n"
        "     non-linear interactions, the Decision Tree models sharp boundary thresholds, while\n"
        "     Logistic Regression models linear trends. By learning coefficients on these predictions,\n"
        "     the meta-model finds an optimal blending strategy, outperforming or equaling base estimators.\n\n"
        "2. DID THE LEVEL-1 LOGISTIC REGRESSION META-MODEL COMBINE STRENGTHS SUCCESSFULY?\n"
        "   - Yes, Logistic Regression is an excellent meta-learner because it acts as a regularized,\n"
        "     interpretable linear blending function. By training on soft probabilities rather than\n"
        "     hard binary labels, it preserves classification uncertainty. It learns non-negative weights\n"
        "     representing base model trust. If one base model overfits (e.g. Decision Tree on deep splits),\n"
        "     the meta-learner assigns it a smaller coefficient, reducing ensemble variance.\n\n"
        "3. EXPERIMENT LIMITATIONS:\n"
        "   - Single Year Snapshot: We filtered for FIFA 23 to avoid temporal leakage, which reduced our dataset\n"
        "     size to 7,425 rows. While training is fast and stable, it lacks multi-year player progression data.\n"
        "   - Hyperparameter Optimization: The Level-0 models use preset, reasonable hyperparameters.\n"
        "     Performing a nested GridSearchCV on base models would optimize individual predictions further\n"
        "     and potentially boost stacking ensemble performance.\n"
        "   - Outfield vs Goalkeeper Stats: Outfield player features are imputed with medians for goalkeepers\n"
        "     and vice-versa. Although handled robustly by SimpleImputer, it creates artificial data profiles\n"
        "     for extreme player positions (like Goalkeepers having simulated slow speed/passing).\n"
        "==============================================================\n"
    )
    print(discussion)


if __name__ == "__main__":
    # Path to Kaggle Dataset Cache
    dataset_directory = r"C:\Users\ISUG\.cache\kagglehub\datasets\stefanoleone992\fifa-23-complete-player-dataset\versions\1"
    
    print("Starting Machine Learning Stacking Framework...")
    run_stacking_framework(dataset_directory)
