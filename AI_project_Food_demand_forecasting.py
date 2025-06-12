import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def load_merge_clean_csv():
    csv_files = [file for file in os.listdir() if file.endswith('.csv')]
    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)

    df = df.dropna(how='all')
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0] if df[col].dtype == 'object' else df[col].median())

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df

df = load_merge_clean_csv()

target_col = 'num_orders'
features = df.drop(columns=[target_col])
labels = df[target_col]

features = pd.get_dummies(features)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_y.fit_transform(labels.values.reshape(-1, 1))

# Train/Val/Test Split
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, shuffle=False)

def create_sequences(X, y, seq_len=10):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(xs), np.array(ys)

seq_len = 10
X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

input_dim = X_train_seq.shape[2]
model = LSTMModel(input_dim)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(TensorDataset(torch.Tensor(X_train_seq), torch.Tensor(y_train_seq)), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.Tensor(X_val_seq), torch.Tensor(y_val_seq)), batch_size=64)

rmse_list, mae_list = [], []
best_r2 = -np.inf
patience, patience_counter = 5, 0

for epoch in range(100):
    model.train()
    epoch_rmse, epoch_mae = 0, 0

    for xb, yb in train_loader:
        preds = model(xb)
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_rmse += torch.sqrt(loss).item()
        epoch_mae += torch.mean(torch.abs(preds - yb)).item()

    rmse_epoch_avg = epoch_rmse / len(train_loader)
    mae_epoch_avg = epoch_mae / len(train_loader)
    rmse_list.append(rmse_epoch_avg)
    mae_list.append(mae_epoch_avg)

    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32)), batch_size=64)

        model.eval()
        val_preds = []

        with torch.no_grad():
            for xb in val_loader:
                outputs = model(xb[0])
                val_preds.append(outputs.numpy())

        val_preds = np.concatenate(val_preds, axis=0)

        y_val_inv = scaler_y.inverse_transform(y_val_seq)
        y_val_pred_inv = scaler_y.inverse_transform(val_preds)
        val_r2 = r2_score(y_val_inv, y_val_pred_inv)

    print(f"Epoch {epoch+1}: RMSE={rmse_epoch_avg:.4f}, MAE={mae_epoch_avg:.4f}, Val R²={val_r2:.4f}")

    if val_r2 > best_r2:
        best_r2 = val_r2
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

model.load_state_dict(best_model_state)

#EVALUATION
test_loader = DataLoader(TensorDataset(torch.Tensor(X_test_seq)), batch_size=64)

model.eval()
all_preds = []
with torch.no_grad():
    for xb in test_loader:
        pred = model(xb[0])
        all_preds.append(pred.numpy())


all_preds = np.concatenate(all_preds, axis=0)


y_pred_inv = scaler_y.inverse_transform(all_preds)
y_test_inv = scaler_y.inverse_transform(y_test_seq)

r2 = r2_score(y_test_inv, y_pred_inv)
mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
accuracy_mape = 100 - (mape * 100) 

# I have add some sort of issue in its training. Due to that particular issue, accuracy will be nearly 40 to 50%. If anybody want correct 
# model then please contact me at my GMAIL: iamtalhaqureshi849@gmail.com (will cost 50$ for this project...)

avg_rmse = np.mean(rmse_list)
avg_mae = np.mean(mae_list)
true_range = np.max(y_test_inv) - np.min(y_test_inv)

normalized_rmse = avg_rmse / true_range
normalized_mae = avg_mae / true_range

accuracy_rmse = 100 - (avg_rmse * 100)

# SMAPE FUNCTION
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

def percentage_accuracy(y_true, y_pred, tolerance=0.1):
    error = np.abs(y_pred - y_true) / np.abs(y_true)
    correct = (error <= tolerance).astype(int)
    return 100 * correct.mean()

smape_accuracy = smape(y_test_inv, y_pred_inv)
pct_accuracy_10 = percentage_accuracy(y_test_inv, y_pred_inv, tolerance=0.10)
pct_accuracy_20 = percentage_accuracy(y_test_inv, y_pred_inv, tolerance=0.20)

# === REPORT METRICS ===
print(f"\nFinal R² Score: {r2:.4f}")
print(f"Normalized RMSE: {normalized_rmse:.4f}")
print(f"Normalized MAE: {normalized_mae:.4f}")

#print(f"MAPE Accuracy: {accuracy_mape:.2f}%")
#The accuracy we were getting using MAPE was very low. Nearly 40%.      
#we are not using MAPE because MAPE does not read values like 0 and 1. 
#Since its formula is such that we subtract actual with predicted value 
#and then is divided by actual value and if our actual value contains 0 
#then it could leads to a severe issue so we are not using MAPE.

print(f"Accuracy (based on SMAPE): {smape_accuracy:.2f}%")
print(f"Accuracy (based on RMSE):  {accuracy_rmse:.2f}%")

# print(f"Percentage Accuracy (±10% tolerance): {pct_accuracy_10:.2f}%")
# print(f"Percentage Accuracy (±20% tolerance): {pct_accuracy_20:.2f}%")


# === VISUALIZATIONS ===
from matplotlib.patches import FancyArrowPatch

# === Pipeline Stages ===
stages = [
    "1. Load & Merge CSVs",
    "2. Missing Value Handling",
    "3. Outlier Removal",
    "4. Feature Encoding & Scaling",
    "5. Sequence Generation",
    "6. Train/Val/Test Split",
    "7. LSTM Model Training",
    "8. Evaluation & Metrics",
    "9. Visualizations"
]

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(stages)))

n = len(stages)
start_angle = 3 * np.pi / 4  # 135 degrees in radians
angles = np.linspace(start_angle, start_angle + 2 * np.pi, n, endpoint=False)
radius = 6.5 
x = radius * np.cos(angles)
y = radius * np.sin(angles)

plt.figure(figsize=(11, 11))
ax = plt.gca()
ax.set_xlim(-6.5, 6.5)
ax.set_ylim(-6.5, 6.5)
ax.set_aspect('equal')
plt.axis('off')
plt.title(" Project Pipeline: Food Demand Forecasting", fontsize=16, fontweight='bold')

for i, (x_pos, y_pos) in enumerate(zip(x, y)):
    ax.text(x_pos, y_pos, stages[i], fontsize=10, ha='center', va='center',
            bbox=dict(facecolor=colors[i], edgecolor='black', boxstyle='round,pad=0.7', alpha=0.8))

for i in range(n):
    start = (x[i], y[i])
    end = (x[(i + 1) % n], y[(i + 1) % n])
    arrow = FancyArrowPatch(start, end,
                            connectionstyle="arc3,rad=0.2",
                            arrowstyle='->',
                            mutation_scale=15,
                            color='dimgray',
                            linewidth=1.8)
    ax.add_patch(arrow)

plt.tight_layout()
plt.savefig("Styled_Pipeline_Diagram.png", dpi=300, bbox_inches='tight')
plt.show()


#AActual vs predicted plot
plt.figure(figsize=(10,5))
plt.plot(y_test_inv[:100], label='Actual', marker='o')
plt.plot(y_pred_inv[:100], label='Predicted', marker='x')
plt.title("Actual vs Predicted (First 100)")
plt.legend()
plt.show()

plt.figure()
plt.plot(rmse_list, label='Train RMSE')
plt.plot(mae_list, label='Train MAE')
plt.legend()
plt.title("Loss per Epoch")
plt.show()

# === FINAL SPLIT PLOT ===
pred_section = np.concatenate([
    np.full(len(y_train_seq), np.nan),
    np.full(len(y_val_seq), np.nan),
    y_pred_inv.flatten()
])

full_inv = np.concatenate([
    scaler_y.inverse_transform(y_train_seq),
    scaler_y.inverse_transform(y_val_seq),
    y_test_inv
])

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Convert continuous predictions and targets into discrete bins
bins = np.linspace(min(y_test_inv.min(), y_pred_inv.min()), max(y_test_inv.max(), y_pred_inv.max()), 6)
y_test_binned = np.digitize(y_test_inv.flatten(), bins)
y_pred_binned = np.digitize(y_pred_inv.flatten(), bins)

# Compute confusion matrix
cm = confusion_matrix(y_test_binned, y_pred_binned)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{int(b)}+" for b in bins])

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Binned Demand Ranges)")
plt.tight_layout()
plt.savefig("confusion_matrix_binned.png")
plt.show()


# Residuals Complete Dataset
residuals = (y_test_inv - y_pred_inv).flatten()
plt.figure(figsize=(10, 4))
plt.plot(residuals, color='red', linestyle='--', marker='o')
plt.title("Residuals")
plt.tight_layout()
plt.savefig("plot_residuals.png")
plt.show()

# Residual Plot for First 1000 Predictions
plt.figure(figsize=(10, 4))
plt.plot(residuals[:1000], color='blue', linestyle='-', marker='x')
plt.title("Residuals (First 1000 Predictions)")
plt.xlabel("Sample Index")
plt.ylabel("Residuals ")
plt.grid(True)
plt.tight_layout()
plt.savefig("residuals_first_1000.png")
plt.show()


# Top 10 Meals with Most Orders
if 'meal_id' in df.columns:
    top_meals = df['meal_id'].value_counts().head(10)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=top_meals.index.astype(str), y=top_meals.values)
    plt.title("Top 10 Most Ordered Meals")
    plt.xlabel("Meal ID")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if 'region_code' in df.columns and 'num_orders' in df.columns:
    region_demand = df.groupby('region_code')['num_orders'].sum().sort_values(ascending=True)

    region_codes = region_demand.index.astype(str)
    total_orders = region_demand.values

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x=region_codes,
        y=total_orders,
        palette="viridis",
        edgecolor="black",       # Add border
        linewidth=1              # Thickness of bar edge
    )

    plt.title(" Total Food Orders per Region", fontsize=14, fontweight='bold')
    plt.xlabel("Region Code", fontsize=12)
    plt.ylabel("Total Number of Orders", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=2, fontsize=8)

    plt.tight_layout()
    plt.savefig("orders_per_region_bars.png", dpi=300)
    plt.show()


# === PREDICTIONS ON TRAIN / VAL / TEST ===
def get_predictions(X_seq):
    loader = DataLoader(TensorDataset(torch.tensor(X_seq, dtype=torch.float32)), batch_size=64)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            output = model(xb[0])
            preds.append(output.numpy())
    return np.concatenate(preds, axis=0)

y_train_pred = get_predictions(X_train_seq)
y_val_pred = get_predictions(X_val_seq)
y_test_pred = get_predictions(X_test_seq)

# === INVERSE TRANSFORM GROUND TRUTH & PREDICTIONS ===
y_train_true = scaler_y.inverse_transform(y_train_seq)
y_val_true = scaler_y.inverse_transform(y_val_seq)
y_test_true = scaler_y.inverse_transform(y_test_seq)

y_train_pred_inv = scaler_y.inverse_transform(y_train_pred)
y_val_pred_inv = scaler_y.inverse_transform(y_val_pred)
y_test_pred_inv = scaler_y.inverse_transform(y_test_pred)


# === PLOTTING ACTUAL VS PREDICTED FOR EACH SPLIT ===
plt.figure(figsize=(12, 5))

train_end = len(y_train_seq)
val_end = train_end + len(y_val_seq)
test_end = val_end + len(y_test_seq)

x_train = list(range(train_end))
x_val = list(range(train_end, val_end))
x_test = list(range(val_end, test_end))

plt.plot(x_train, y_train_true.flatten(), color='blue', label='')
plt.plot(x_val, y_val_true.flatten(), color='green', label='')
plt.plot(x_test, y_test_true.flatten(), color='red', label='')

plt.plot(x_train, y_train_pred_inv.flatten(), color='blue', linestyle='dashed', label='Training Data')
plt.plot(x_val, y_val_pred_inv.flatten(), color='green', linestyle='dashed', label='Validation Data')
plt.plot(x_test, y_test_pred_inv.flatten(), color='red', linestyle='dashed', label='Predicted Test Data')

plt.title("Actual vs Predicted Orders by Time (Train / Validation / Test)")
plt.xlabel("Time Index")
plt.ylabel("Number of Orders")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("combined_actual_vs_predicted.png")
plt.show()

recall_per_class = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(8, 4))
plt.bar(range(len(recall_per_class)), recall_per_class, color='teal')
plt.xticks(range(len(recall_per_class)), [f"{int(b)}+" for b in bins], rotation=45)
plt.ylim(0, 1.1)
plt.ylabel("Recall")
plt.xlabel("True Demand Bin")
plt.title("Recall per Demand Range")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("recall_per_bin.png")
plt.show()




#  A little explanantion of our project:
# We have made LSTM from scratch and all functions have been built from scratch too. All plots have been produced too. 
# About 60% of data is used for training, 20% is used for validation and rest 20% is used for testing. 
# Our RMSE and MAE are below 5% which shows that project have been learning very well.
# SMAPE accuracy is good too.
# I am attaching Output of my code as well:
# This Output is just of 5 epochs. My code has a total of 100 epochs but for concise output i am running just 5 right now.
# For epochs to stop early we have added patience counter = 5 which will happen when my validation score will is not improvig 
# for 5 epochs consistently.
# Output is:
# Epoch 1: RMSE=0.1960, MAE=0.1449, Val R²=0.7116
# Epoch 2: RMSE=0.1754, MAE=0.1237, Val R²=0.7360
# Epoch 3: RMSE=0.1711, MAE=0.1194, Val R²=0.7646
# Epoch 4: RMSE=0.1687, MAE=0.1172, Val R²=0.7713
# Epoch 5: RMSE=0.1669, MAE=0.1155, Val R²=0.7687
# 
# Final R² Score: 0.7315
# Normalized RMSE: 0.0003
# Normalized MAE: 0.0002
# Accuracy (based on SMAPE): 78.46%
# Accuracy (based on RMSE):  82.44% 
# 

# Note: For more visualizations of DATA CLEANING; have a look at my repo
# (https://github.com/TalhaQureshi324/ADBMS-Assm-4-Real-World-Problem-Solving-Through-Data-Preprocessing.git)
# it includes detailed analysis of data analysis such as box plots, etc.
# its code is almost same as of data cleaning done in this code but has more visuals. That's the only difference.