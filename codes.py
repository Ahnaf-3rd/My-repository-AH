import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# ============================================================
# 1. Load and Prepare Training Data
# ============================================================

# Load the single modern bird melanosome dataset (already in nm)
train = pd.read_csv("modern_bird_melanosomes_with_source.csv")

# Keep only the columns we need and standardize spelling
train = train[["colour", "length_nm", "width_nm", "aspect_ratio"]].copy()
train.rename(columns={"colour": "color"}, inplace=True)
train["color"] = train["color"].str.lower().replace("grey", "gray")

# Exclude non-dinosaur specific classes (e.g., penguin)
train = train[train["color"] != "penguin"].reset_index(drop=True)

print(f"Total training samples after cleanup: {len(train)}")
print("Class distribution:\n", train["color"].value_counts())

# ============================================================
# 2. Load Validation (Theropods) & Test Sets (Non-Theropods)
# ============================================================
val = pd.read_csv("theropod_validation.csv")
val.rename(columns={"known_colour": "known_color"}, inplace=True)
val["known_color"] = val["known_color"].str.lower().replace("grey", "gray")

test = pd.read_csv("non_theropod_test.csv")

# ============================================================
# 3. Preprocessing & Feature Selection
# ============================================================
# Using aspect_ratio as primary feature due to taphonomic distortion in fossils
features = ["aspect_ratio"]

X_train = train[features]
y_train = train["color"]

# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Apply SMOTE to handle class imbalance. Set k_neighbors=2 since the 'yellow' class has only 3 samples.
smote = SMOTE(k_neighbors=2, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train_enc)

# Prepare Validation and Test sets
X_val = val[features]
y_val_enc = le.transform(val["known_color"])
X_val_scaled = scaler.transform(X_val)

X_test = test[features]
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. Model Training & Evaluation
# ============================================================
models = {
    "QDA": QuadraticDiscriminantAnalysis(),
    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        class_weight="balanced",
        probability=True,
        random_state=42,
    ),
    "RF": RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        class_weight="balanced_subsample",
        random_state=42,
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n{'='*50}\nMODEL: {name}\n{'='*50}")

    # Train model on balanced data
    model.fit(X_train_bal, y_train_bal)

    # Cross-validation score
    cv_scores = cross_val_score(
        model, X_train_bal, y_train_bal, cv=cv, scoring="accuracy"
    )
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Validate on Theropod fossils
    val_pred = model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val_enc, val_pred)
    print(f"Theropod Validation Accuracy: {val_acc:.3f}")

    print("\nIndividual Theropod Predictions:")
    for i, (_, row) in enumerate(val.iterrows()):
        true_color = row["known_color"]
        pred_color = le.inverse_transform([val_pred[i]])[0]
        correct = "✓" if true_color == pred_color else "✗"
        print(
            f"  {row['species']:25s} | True: {true_color:10s} | Pred: {pred_color:10s} | {correct}"
        )

    # Predict on Non-Theropod test set
    test_pred = model.predict(X_test_scaled)
    test_proba = model.predict_proba(X_test_scaled)

    print("\nNon-Theropod Predictions:")
    for i, (_, row) in enumerate(test.iterrows()):
        pred = le.inverse_transform([test_pred[i]])[0]
        prob = max(test_proba[i])
        print(f"  {row['species']}: {pred} (Confidence = {prob:.2f})")

# ============================================================
# 5. Visualizations
# ============================================================

# Figure 1: Boxplot of Aspect Ratio by Color Class
plot_data = train[
    train["color"].isin(["black", "brown", "gray", "iridescent"])
].copy()
order = (
    plot_data.groupby("color")["aspect_ratio"]
    .median()
    .sort_values()
    .index
)

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=plot_data,
    x="color",
    y="aspect_ratio",
    order=order,
    hue="color",
    palette="Set2",
    legend=False,
    showfliers=False,
)
sns.stripplot(
    data=plot_data,
    x="color",
    y="aspect_ratio",
    order=order,
    color="black",
    alpha=0.3,
    size=2,
)

plt.ylabel("Aspect Ratio (Length / Width)", fontsize=12)
plt.xlabel("Color Class", fontsize=12)
plt.title("Melanosome Aspect Ratio by Color in Modern Birds", fontsize=14)
plt.tight_layout()
plt.savefig("figure1_boxplot.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 2: Morphospace Distribution Shift
train_plot = plot_data.copy()
train_plot["log_length"] = np.log10(train_plot["length_nm"])
train_plot["log_width"] = np.log10(train_plot["width_nm"])

test_plot = test.copy()
test_plot["log_length"] = np.log10(test_plot["length_nm"])
test_plot["log_width"] = np.log10(test_plot["width_nm"])
test_plot["genus"] = test_plot["species"].str.split().str[0]

color_palette = {
    "black": "#2c3e50",
    "brown": "#8B4513",
    "gray": "#7f8c8d",
    "iridescent": "#1abc9c",
}

fig, ax = plt.subplots(figsize=(8, 6))

for color, group in train_plot.groupby("color"):
    ax.scatter(
        group["log_length"],
        group["log_width"],
        label=color.capitalize(),
        color=color_palette[color],
        alpha=0.5,
        s=20,
        edgecolor="none",
    )

marker_map = {"Psittacosaurus": "*", "Diplodocus": "D"}
size_map = {"Psittacosaurus": 500, "Diplodocus": 120}

for genus, group in test_plot.groupby("genus"):
    ax.scatter(
        group["log_length"],
        group["log_width"],
        marker=marker_map.get(genus, "o"),
        s=size_map.get(genus, 200),
        color="red",
        edgecolor="black",
        linewidth=0.8,
        label=genus,
    )

ax.set_xlabel("log₁₀(Length / nm)", fontsize=12)
ax.set_ylabel("log₁₀(Width / nm)", fontsize=12)
ax.set_title("Melanosome Morphospace: Training vs. Non-Theropod Fossils", fontsize=14)
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("figure2_distribution_shift.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nPipeline complete. Figures saved as 'figure1_boxplot.png' and 'figure2_distribution_shift.png'.")
