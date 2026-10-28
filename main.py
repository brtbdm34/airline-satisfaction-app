import os
import kagglehub
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score

print("=== 1. VERİ SETİNİN İNDİRİLMESİ VE ÖN İŞLEME ===")
path = kagglehub.dataset_download("teejmahal20/airline-passenger-satisfaction")

train_df = pd.read_csv(os.path.join(path, "train.csv"))
test_df = pd.read_csv(os.path.join(path, "test.csv"))

drop_cols = ['Unnamed: 0', 'id']
train_df = train_df.drop(columns=drop_cols, errors='ignore')
test_df = test_df.drop(columns=drop_cols, errors='ignore')

# Eksik verileri medyan ile doldurma
train_df['Arrival Delay in Minutes'] = train_df['Arrival Delay in Minutes'].fillna(train_df['Arrival Delay in Minutes'].median())
test_df['Arrival Delay in Minutes'] = test_df['Arrival Delay in Minutes'].fillna(test_df['Arrival Delay in Minutes'].median())

# Kategorik sütunları kodlama
cat_cols = train_df.select_dtypes(include=['object']).columns
for col in cat_cols:
    train_df[col] = train_df[col].astype('category').cat.codes
    test_df[col] = test_df[col].astype('category').cat.codes

X_train = train_df.drop(columns=['satisfaction'])
y_train = train_df['satisfaction']
X_test = test_df.drop(columns=['satisfaction'])
y_test = test_df['satisfaction']

print("\n=== 2. MODEL EĞİTİMİ VE KARŞILAŞTIRMA ===")

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)

xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

print(f"1. Decision Tree       : %{dt_acc * 100:.2f}")
print(f"2. Random Forest       : %{rf_acc * 100:.2f}")
print(f"3. Gradient Boosting   : %{gb_acc * 100:.2f}")
print(f"4. XGBoost             : %{xgb_acc * 100:.2f}")

models_data = {
    'Decision Tree': {'model': dt_model, 'pred': dt_pred, 'acc': dt_acc},
    'Random Forest': {'model': rf_model, 'pred': rf_pred, 'acc': rf_acc},
    'Gradient Boosting': {'model': gb_model, 'pred': gb_pred, 'acc': gb_acc},
    'XGBoost': {'model': xgb_model, 'pred': xgb_pred, 'acc': xgb_acc}
}

best_model_name = max(models_data, key=lambda x: models_data[x]['acc'])
best_model = models_data[best_model_name]['model']

print(f"\n🏆 Şampiyon Model Seçildi: {best_model_name}")

print("\n=== 3. K-FOLD CROSS-VALIDATION ===")
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"{best_model_name} 5-Fold Ortalama Doğruluk: %{cv_scores.mean() * 100:.2f}")

print("\n=== 4. ROC-AUC EĞRİSİ HESAPLAMA ===")
best_probs = best_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, best_probs)
print(f"ROC-AUC Skoru: {auc_score:.4f}")

fpr, tpr, thresholds = roc_curve(y_test, best_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'{best_model_name} - ROC Eğrisi')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_auc_curve.png')
plt.close()
print("ROC eğrisi kaydedildi.")

print("\n=== 5. SHAP ANALİZİ ===")
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test.iloc[:500])

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test.iloc[:500], show=False)
plt.title(f"{best_model_name} - SHAP Özellik Önem Analizi", fontsize=12)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("SHAP özet grafiği kaydedildi.")

print("\n=== 6. MODELİN KAYDEDİLMESİ (.pkl) ===")
model_info = {
    'all_models': models_data,
    'name': best_model_name,
    'model': best_model
}
joblib.dump(model_info, 'best_airline_model.pkl')
print("Model başarıyla 'best_airline_model.pkl' olarak kaydedildi!")