import os
import time
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Suppress minor sklearn warning logs
warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- Model Training and Data Pipeline Setup ---
print("Initializing Machine Learning Models for Stacking Dashboard...")

dataset_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / "stefanoleone992" / "fifa-23-complete-player-dataset" / "versions" / "1"
path = Path(dataset_dir)
csv_files = list(path.glob("*.csv"))

# Select player files that are under 500 MB
player_files = [
    f for f in csv_files
    if "player" in f.name.lower()
    and "coach" not in f.name.lower()
    and "team" not in f.name.lower()
]
manageable_files = [f for f in player_files if f.stat().st_size < 500 * 1024 * 1024]
selected_file = max(manageable_files, key=lambda f: f.stat().st_size)

print(f"Loading dataset: {selected_file.name}")
df = pd.read_csv(selected_file, low_memory=False)

if 'fifa_version' in df.columns:
    df = df[df['fifa_version'] == 23].copy()

# Features structure
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
categorical_cols = ['preferred_foot', 'work_rate', 'body_type']

X = df[numeric_cols + categorical_cols]
y = (df['overall'] >= 75).astype(int)

# Split and preprocess
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
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

# Compute default profiles to fill in unsubmitted columns in UI
default_profile = {}
for col in numeric_cols:
    default_profile[col] = float(X_train[col].median())
for col in categorical_cols:
    default_profile[col] = str(X_train[col].mode()[0])

base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Perceptron": CalibratedClassifierCV(estimator=Perceptron(random_state=42)),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6)
}

# Define Out-of-Fold generation function
def generate_oof_predictions(models, X_train, y_train, preprocessor, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = {name: np.zeros(len(X_train)) for name in models}
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        fold_preprocessor = clone(preprocessor)
        X_tr_proc = fold_preprocessor.fit_transform(X_tr)
        X_val_proc = fold_preprocessor.transform(X_val)
        
        for name, model in models.items():
            fold_model = clone(model)
            fold_model.fit(X_tr_proc, y_tr)
            oof_preds[name][val_idx] = fold_model.predict_proba(X_val_proc)[:, 1]
            
    final_preprocessor = clone(preprocessor)
    X_train_proc = final_preprocessor.fit_transform(X_train)
    
    trained_models = {}
    for name, model in models.items():
        full_model = clone(model)
        full_model.fit(X_train_proc, y_train)
        trained_models[name] = full_model
        
    return pd.DataFrame(oof_preds), trained_models, final_preprocessor

# Generate OOF and train meta model
oof_df, trained_base_models, final_preprocessor = generate_oof_predictions(
    base_models, X_train, y_train, preprocessor, n_splits=5
)
meta_model = LogisticRegression(random_state=42)
meta_model.fit(oof_df, y_train)
print("Models successfully initialized and trained in Flask memory!")


# --- Flask Server Routes ---

@app.route('/')
def home():
    # Render index.html template from templates folder
    try:
        with open('templates/index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return "Frontend HTML not found. Make sure templates/index.html is created.", 404


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Start with default player profile and overwrite with sliders
        player = default_profile.copy()
        for k, v in data.items():
            if k in player:
                if k in numeric_cols:
                    player[k] = float(v)
                else:
                    player[k] = str(v)
                    
        # Perform feature adjustments for consistency
        # e.g., if weak_foot / skill_moves are updated
        df_player = pd.DataFrame([player])
        X_proc = final_preprocessor.transform(df_player)
        
        # Get base model predicted probabilities
        lr_prob = trained_base_models["Logistic Regression"].predict_proba(X_proc)[0, 1]
        pc_prob = trained_base_models["Perceptron"].predict_proba(X_proc)[0, 1]
        dt_prob = trained_base_models["Decision Tree"].predict_proba(X_proc)[0, 1]
        
        selected_model = data.get("model", "Stacking Ensemble")
        
        if selected_model == "Logistic Regression":
            prob = lr_prob
        elif selected_model == "Perceptron":
            prob = pc_prob
        elif selected_model == "Decision Tree":
            prob = dt_prob
        else:  # Stacking Ensemble
            meta_features = pd.DataFrame([{
                "Logistic Regression": lr_prob,
                "Perceptron": pc_prob,
                "Decision Tree": dt_prob
            }])
            prob = meta_model.predict_proba(meta_features)[0, 1]
            
        is_elite = int(prob >= 0.5)
        
        return jsonify({
            "success": True,
            "probability": float(prob),
            "is_elite": is_elite,
            "base_probabilities": {
                "Logistic Regression": float(lr_prob),
                "Perceptron": float(pc_prob),
                "Decision Tree": float(dt_prob)
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/plots/<path:filename>')
def serve_plot(filename):
    # Serve generated comparison figures from plots/ directory
    plots_dir = os.path.join(app.root_path, 'plots')
    return send_from_directory(plots_dir, filename)


if __name__ == '__main__':
    print("Starting Flask dashboard server on http://localhost:8080")
    app.run(debug=False, port=8080)
