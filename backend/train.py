import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib  # to push this model to backend, or pickle also we can use

df = pd.read_csv('../data/car_resale_data.csv')
df
df.head()
df.describe()
df.corr()
df.dtypes
(df == 0).sum()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.savefig('correlation_heatmap.png')
plt.show()

x = df.drop('Resale_Price_USD', axis=1)
y = df['Resale_Price_USD']
feature_names = list(x.columns)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Lasso needs scaled features (L1 penalty), and the point of using Lasso here
# specifically is feature selection: this dataset has 3 irrelevant/noise
# columns (Color_Popularity_Score, Local_Fuel_Price_Index,
# Service_Center_Visits) mixed in with 5 real signal columns. Lasso should
# shrink the irrelevant ones toward/at zero.
model = make_pipeline(StandardScaler(), Lasso(alpha=50.0, max_iter=10000))
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

coefs = model.named_steps['lasso'].coef_
importance = dict(zip(feature_names, [round(float(c), 2) for c in coefs]))
print("Coefficients (scaled):", importance)

with open('feature_importance.json', 'w') as f:
    json.dump(importance, f, indent=2)

joblib.dump(model, 'model_lasso.pkl')

print('Model Trained and Saved')
