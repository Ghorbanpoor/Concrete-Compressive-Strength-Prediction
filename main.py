#!/usr/bin/env python
# coding: utf-8

"""
پیش‌بینی مقاومت فشاری بتن (Concrete Compressive Strength)
قابل اجرا در GitHub Codespaces
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import itertools as it

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ۱. بارگذاری دیتاست
# ============================================================
df = pd.read_csv('concrete_data.csv', sep=',')
print("=" * 60)
print("نمای اولیه دیتاست:")
print(df.head(5))

# ============================================================
# ۲. ساختار داده
# ============================================================
print("\n" + "=" * 60)
print(f"تعداد ردیف‌ها: {df.shape[0]}")
print(f"تعداد ستون‌ها: {df.shape[1]}")
print(df.info())

# ============================================================
# ۳. مقادیر گمشده
# ============================================================
print("\n" + "=" * 60)
print("مقادیر گمشده:")
print(df.isnull().sum())
print("دیتاست هیچ مقدار گمشده‌ای ندارد.")

# ============================================================
# ۴. تجسم داده‌ها
# ============================================================
print("\n" + "=" * 60)
print("در حال رسم نمودارها...")

# ۴.۱ ماتریس همبستگی
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, linewidth=2, cmap='coolwarm')
plt.title("همبستگی بین متغیرها")
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150)
plt.show()

# ۴.۲ نمودار جفتی
sns.pairplot(df, markers="h")
plt.savefig('pairplot.png', dpi=150)
plt.show()

# ۴.۳ توزیع مقاومت بتن
plt.figure(figsize=(8, 5))
sns.histplot(df['concrete_compressive_strength'], bins=10, color='b', kde=True)
plt.ylabel("فراوانی")
plt.title("توزیع مقاومت بتن")
plt.tight_layout()
plt.savefig('distribution.png', dpi=150)
plt.show()

# ۴.۴ توزیع اجزای بتن
cols = [i for i in df.columns if i != 'concrete_compressive_strength']
length = len(cols)
cs = ["b", "r", "g", "c", "m", "k", "lime", "orange"]
fig = plt.figure(figsize=(13, 25))

for i, j, k in it.zip_longest(cols, range(length), cs):
    plt.subplot(4, 2, j + 1)
    sns.histplot(df[i], color=k, kde=True)
    plt.axvline(df[i].mean(), linestyle="dashed", label="میانگین", color="k")
    plt.legend(loc="best")
    plt.title(i, color="navy")
    plt.xlabel("")

plt.tight_layout()
plt.savefig('components_distribution.png', dpi=150)
plt.show()

# ۴.۵ نمودار پراکندگی سیمان در برابر آب
fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(111)
sc = plt.scatter(
    df["water"], df["cement"],
    c=df["concrete_compressive_strength"],
    s=df["concrete_compressive_strength"] * 3,
    linewidth=1, edgecolor="k", cmap="viridis"
)
ax.set_xlabel("آب")
ax.set_ylabel("سیمان")
lab = plt.colorbar(sc)
lab.set_label("مقاومت فشاری بتن")
plt.title("سیمان در برابر آب")
plt.tight_layout()
plt.savefig('cement_vs_water.png', dpi=150)
plt.show()

# ============================================================
# ۵. تقسیم داده‌ها (۷۰٪ آموزش، ۳۰٪ تست)
# ============================================================
train, test = train_test_split(df, test_size=0.3, random_state=0)

feature_cols = [x for x in df.columns if x not in ["concrete_compressive_strength", "age_months"]]

train_X = train[feature_cols]
train_Y = train["concrete_compressive_strength"]
test_X = test[feature_cols]
test_Y = test["concrete_compressive_strength"]

print("\n" + "=" * 60)
print(f"ابعاد داده آموزش: {train_X.shape}")
print(f"ابعاد داده تست: {test_X.shape}")

# ============================================================
# ۶. تعریف توابع کمکی
# ============================================================
def evaluate_model(model_name, y_true, y_pred, model=None, X_test=None):
    """ارزیابی عملکرد مدل"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    r2 = None
    if model is not None and X_test is not None:
        r2 = model.score(X_test, y_true)
    
    print(f"\n📊 مدل: {model_name}")
    print(f"   R² (دقت): {r2 if r2 is not None else 'N/A':.4f}" if r2 else f"   R² (دقت): N/A")
    print(f"   MAE: {mae:.4f}")
    print(f"   MSE: {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    return {"model": model_name, "R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}


def plot_actual_vs_predicted(y_true, y_pred, title, filename):
    """رسم نمودار مقادیر واقعی در برابر پیش‌بینی‌شده"""
    dat = pd.DataFrame({'واقعی': y_true, 'پیش‌بینی': y_pred})
    dat1 = dat.head(25)
    dat1.plot(kind='bar', figsize=(10, 6))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_feature_importance(coefs, feature_names, title, filename):
    """رسم اهمیت ویژگی‌ها"""
    colors = cm.rainbow(np.linspace(0, 1, len(feature_names)))
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, coefs, color=colors)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


# ============================================================
# ۷. مدل‌ها
# ============================================================
results = []

# --- مدل ۱: رگرسیون خطی چندگانه ---
print("\n" + "=" * 60)
print("مدل ۱: رگرسیون خطی چندگانه")
lm = LinearRegression()
model1 = lm.fit(train_X, train_Y)
predictions1 = lm.predict(test_X)
results.append(evaluate_model("Linear", test_Y, predictions1, model1, test_X))

plot_feature_importance(
    lm.coef_.ravel(), feature_cols,
    "اهمیت ویژگی‌ها - رگرسیون خطی",
    "linear_feature_importance.png"
)
plot_actual_vs_predicted(
    test_Y, predictions1,
    "مقادیر واقعی در برابر پیش‌بینی - رگرسیون خطی",
    "linear_actual_vs_pred.png"
)

# --- مدل ۲: LASSO ---
print("\n" + "=" * 60)
print("مدل ۲: رگرسیون LASSO")
las = Lasso(alpha=0.1)
model2 = las.fit(train_X, train_Y)
predictions2 = las.predict(test_X)
results.append(evaluate_model("LASSO", test_Y, predictions2, model2, test_X))

plot_feature_importance(
    las.coef_.ravel(), feature_cols,
    "اهمیت ویژگی‌ها - LASSO",
    "lasso_feature_importance.png"
)
plot_actual_vs_predicted(
    test_Y, predictions2,
    "مقادیر واقعی در برابر پیش‌بینی - LASSO",
    "lasso_actual_vs_pred.png"
)

# --- مدل ۳: KNN ---
print("\n" + "=" * 60)
print("مدل ۳: KNN")
knn = KNeighborsRegressor()
model3 = knn.fit(train_X, train_Y)
predictions3 = knn.predict(test_X)
results.append(evaluate_model("KNN", test_Y, predictions3, model3, test_X))

plot_actual_vs_predicted(
    test_Y, predictions3,
    "مقادیر واقعی در برابر پیش‌بینی - KNN",
    "knn_actual_vs_pred.png"
)

# --- مدل ۴: SVM ---
print("\n" + "=" * 60)
print("مدل ۴: SVM")
svm = SVR(kernel='linear')
model4 = svm.fit(train_X, train_Y)
predictions4 = svm.predict(test_X)
results.append(evaluate_model("SVM", test_Y, predictions4, model4, test_X))

plot_actual_vs_predicted(
    test_Y, predictions4,
    "مقادیر واقعی در برابر پیش‌بینی - SVM",
    "svm_actual_vs_pred.png"
)

# --- مدل ۵ و ۶: چندجمله‌ای درجه ۲ و ۳ ---
for degree in [2, 3]:
    print(f"\n" + "=" * 60)
    print(f"مدل {5 if degree == 2 else 6}: چندجمله‌ای درجه {degree}")
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    train_X_poly = poly.fit_transform(train_X)
    test_X_poly = poly.transform(test_X)
    
    reg = LinearRegression()
    model_poly = reg.fit(train_X_poly, train_Y)
    y_pred_poly = reg.predict(test_X_poly)
    
    results.append(evaluate_model(
        f"Poly n={degree}", test_Y, y_pred_poly, model_poly, test_X_poly
    ))
    
    plot_actual_vs_predicted(
        test_Y, y_pred_poly,
        f"مقادیر واقعی در برابر پیش‌بینی - چندجمله‌ای درجه {degree}",
        f"poly{degree}_actual_vs_pred.png"
    )

# ============================================================
# ۸. مقایسه مدل‌ها
# ============================================================
print("\n" + "=" * 60)
print("📈 مقایسه مدل‌ها:")

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# نمودار مقایسه دقت
plt.figure(figsize=(10, 6))
bars = results_df['model'].tolist()
r2_values = results_df['R2'].tolist()
colors = cm.rainbow(np.linspace(0, 1, len(bars)))
plt.bar(bars, r2_values, color=colors)
plt.xlabel('مدل‌ها')
plt.ylabel('دقت (R²)')
plt.title('مقایسه دقت مدل‌ها')
plt.ylim(0, 1)
for i, v in enumerate(r2_values):
    if v is not None:
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.savefig('model_comparison_r2.png', dpi=150)
plt.show()

# نمودار مقایسه RMSE
plt.figure(figsize=(10, 6))
rmse_values = results_df['RMSE'].tolist()
plt.bar(bars, rmse_values, color=colors)
plt.xlabel('مدل‌ها')
plt.ylabel('RMSE')
plt.title('مقایسه RMSE مدل‌ها')
for i, v in enumerate(rmse_values):
    plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
plt.tight_layout()
plt.savefig('model_comparison_rmse.png', dpi=150)
plt.show()

print("\n✅ اجرا کامل شد! همه نمودارها ذخیره شدند.")
