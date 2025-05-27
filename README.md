# 📈 Food Demand Forecasting using LSTM (Time Series Regression)

📚 **Final Project — Artificial Intelligence (BSCS, 4th Semester)**

This project was developed as the final project for the Artificial Intelligence course in the 4th semester of the BSCS program. It focuses on predicting food demand across fulfillment centers using time series analysis and an LSTM (Long Short-Term Memory) model implemented from scratch in PyTorch. The model helps optimize supply chain operations, minimize food wastage, and improve inventory planning.

---

## 🔍 Problem Statement

Accurately forecasting the demand for meals at various fulfillment centers is crucial for reducing overproduction and minimizing operational costs. The goal of this project is to use deep learning (specifically LSTM networks) to build a regression model that learns from historical sales data and predicts future food demand.

---

## 📂 Dataset Description

- **Source**: Kaggle – [Food Demand Forecasting Dataset](https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting)
- **Format**: CSV
- **Size**: ~450,000 records
- **Type**: Tabular time-series data

### 📊 Attributes Used

| Feature               | Description                                  |
|-----------------------|----------------------------------------------|
| `week`                | Week number                                  |
| `center_id`           | Fulfillment center ID                        |
| `meal_id`             | Meal item ID                                 |
| `checkout_price`      | Final price after discounts                  |
| `base_price`          | Original price                               |
| `emailer_for_promotion` | Whether email promotion was sent (0/1)     |
| `homepage_featured`   | Whether meal was featured on homepage (0/1)  |
| `num_orders`          | **Target variable** — number of meals ordered |

---

## ⚙️ Project Pipeline

The complete project flow is:

1. **Data Loading & Merging** – Read and combine all CSVs
2. **Data Cleaning** – Remove duplicates, fill missing values
3. **Outlier Handling** – IQR-based filtering
4. **Feature Encoding & Scaling** – One-hot encoding + MinMaxScaler
5. **Sequence Creation** – Fixed-length time windows (seq_len = 10)
6. **Train/Validation/Test Split** – Chronological 60/20/20 split
7. **LSTM Model Implementation** – Built from scratch using PyTorch
8. **Training Loop** – With early stopping using validation R²
9. **Evaluation & Visualization** – Metrics + plots for insights

📌 *See `Styled_Pipeline_Diagram.png` for visual representation.*

---

## 🧠 LSTM Model Architecture

```python
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        ...
```

- **Input Dim**: Number of features after encoding
- **Hidden Dim**: 128
- **Layers**: 2 LSTM layers
- **Dropout**: 0.2
- **Output**: Single value (regression)

---

## 🏋️‍♂️ Training Details

- **Sequence Length**: 10
- **Batch Size**: 64
- **Optimizer**: Adam
- **Loss Function**: MSELoss
- **Epochs**: 50
- **Early Stopping**: Based on best R² on validation set

---

## 📏 Evaluation Metrics

- **R² Score** (coefficient of determination)
- **RMSE** – Root Mean Square Error
- **MAE** – Mean Absolute Error
- **SMAPE** – Symmetric Mean Absolute Percentage Error
- **Percentage Accuracy** – within ±10% and ±20% tolerance

---

## 📉 Output Metrics (Example)

```txt
Epoch 1: RMSE=0.1960, MAE=0.1449, Val R²=0.7116
...
Final R² Score: 0.7315
Normalized RMSE: 0.0003
Normalized MAE: 0.0002
Accuracy (based on SMAPE): 78.46%
Accuracy (based on RMSE):  82.44%
```

---

## 📊 Visualizations

| Plot                             | Description                                   |
|----------------------------------|-----------------------------------------------|
| `Actual vs Predicted`            | First 100 test points                         |
| `Loss per Epoch`                 | RMSE and MAE over epochs                      |
| `Residual Plots`                 | Error distribution over predictions           |
| `Confusion Matrix (Binned)`      | True vs predicted demand categories           |
| `Recall per Demand Range`        | Class-wise recall from confusion matrix       |
| `Top Ordered Meals`              | Bar plot of top 10 most ordered meal IDs      |
| `Orders per Region`              | Region-wise total food demand                 |

---

## 🛠 Technologies Used

- **Language**: Python
- **Libraries**: PyTorch, NumPy, Pandas, Seaborn, Matplotlib, Scikit-learn
- **Environment**: Jupyter / Python scripts

---

## 🧪 How to Run

1. Clone the repo:
```bash
git clone https://github.com/TalhaQureshi324/Food-Demand-Forecasting-Using-Time-series-analysis-LSTM-model-.git
cd Food-Demand-Forecasting-Using-Time-series-analysis-LSTM-model-
```

2. Place all `.csv` files in the root directory.

3. Run the script:
```bash
python your_main_script.py
```

4. Outputs like metrics and visualizations will be saved in the same folder.

---

## 🧠 Notes

- MAPE was not reliable due to division by zero issues; SMAPE was used instead.
- All LSTM modeling, training, and evaluation was implemented from scratch using PyTorch.
- Data sequences are generated with a window size of 10 for temporal learning.
- Model was evaluated using both regression metrics and classification-style confusion matrix (binned).

---

## ✅ Final Summary

This project delivers a highly optimized deep learning pipeline to predict food demand using historical time series data. The use of LSTM enables the model to capture temporal dependencies, and the results show excellent performance with low RMSE/MAE and high accuracy via SMAPE.

All visualizations and model components were built from scratch, with additional plots to analyze recall, residuals, and prediction confidence.
