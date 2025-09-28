# services/validation_service.py
# ======================================================
# 功能: 使用 k-fold cross-validation 進行模型驗證
# 作者: Silvia 團隊
# ======================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# === 範例數據 (可替換成 docking 分數 或 分子屬性資料) ===
def load_sample_data():
    """
    載入範例資料
    X: 特徵 (隨機產生)
    y: 標籤 (0/1 分類)
    """
    np.random.seed(42)
    X = np.random.rand(100, 5)   # 100 筆數據, 每筆 5 個特徵
    y = np.random.randint(0, 2, 100)  # 0 或 1
    return X, y


# === K-Fold 驗證流程 ===
def run_kfold_validation(X, y, k=5):
    """
    執行 K-Fold 驗證
    參數:
        X: 特徵矩陣
        y: 標籤
        k: 幾折交叉驗證
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    # === 逐折訓練與驗證 ===
    for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 建立模型 (Logistic Regression)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # 驗證
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        print(f"第 {fold} 折準確率: {acc:.4f}")

    print(f"\n平均準確率: {np.mean(accuracies):.4f}")
    return accuracies


if __name__ == "__main__":
    # SOP: 新工程師只要執行 python services/validation_service.py 就能測試流程
    X, y = load_sample_data()
    run_kfold_validation(X, y, k=5)
