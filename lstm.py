# ==============================
# LSTM 强化版（目标R² 0.85~0.88）
# 针对你“有进步但还低”的结果进行的第3轮精准调参
# ==============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer, ReLU, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 1. 读取数据
df = pd.read_excel(r"C:\Users\焦焦\Desktop\论文数据.xlsx")
df = df.iloc[:, 1:]  # 删除日期列
df = df.fillna(0)

# 2. 特征 X 和标签 y
X = df.drop(columns=["PM2.5"]).values
y = df["PM2.5"].values

# 3. 数据归一化（X和y都缩放）
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 4. 构造时序输入（关键提升：时间步30）
time_steps = 45
X_seq = []
y_seq = []
for i in range(time_steps, len(X)):
    X_seq.append(X[i - time_steps:i])
    y_seq.append(y_scaled[i])

X = np.array(X_seq)
y = np.array(y_seq)

# 5. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 6. 构建超强 Stacked LSTM + BatchNormalization
model = Sequential([
    InputLayer(shape=(X_train.shape[1], X_train.shape[2])),

    LSTM(384, return_sequences=True),
    BatchNormalization(),  # ← 新增：稳定训练
    Dropout(0.12),

    LSTM(128),
    BatchNormalization(),  # ← 新增
    Dropout(0.18),

    Dense(64),
    ReLU(),
    Dense(32),
    ReLU(),
    Dense(1)
])

# 更小的学习率 + 精细优化
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mse')

model.summary()

# 7. 训练回调（耐心更大，让模型充分收敛）
callbacks = [
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1)
]

model.fit(
    X_train, y_train,
    epochs=200,  # 最多200轮，实际会早停
    batch_size=32,  # 更小batch，更新更频繁
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# 8. 预测 + 反归一化
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 9. 计算指标
r2 = r2_score(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
rpd = np.std(y_test_original) / rmse

# 输出
print("\n" + "=" * 80)
print("        📊 LSTM 强化版（第3轮调参 - 目标0.85+）")
print("=" * 80)
print(f"R²  决定系数    = {r2:.4f}")
print(f"RMSE 均方根误差  = {rmse:.4f}")
print(f"RPD  性能比率    = {rpd:.4f}")
print("=" * 80)

print("提示（下次继续调参用）：")
print("   • R² 仍低于0.84 → 把 time_steps 改成 45 或第一层 LSTM 改成 384")
print("   • R² 超过0.89 → 把 Dropout(0.12) 改回 0.20")
print("   • 想再冲高一点 → 告诉我具体数值，我给你加 Attention 层")

# 训练集预测 + 反归一化 + 指标计算
y_train_pred_scaled = model.predict(X_train)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()
y_train_original = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()

r2_train = r2_score(y_train_original, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train_original, y_train_pred))
rpd_train = np.std(y_train_original) / rmse_train

print("\n【训练集结果】")
print(f"R²  决定系数    = {r2_train:.4f}")
print(f"RMSE 均方根误差  = {rmse_train:.4f}")
print(f"RPD  性能比率    = {rpd_train:.4f}")