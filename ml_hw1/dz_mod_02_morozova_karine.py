from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)

data = california_housing['frame']

# %%
from scipy.stats import zscore
    
features_of_interest = ['AveRooms', "AveBedrms", 'AveOccup', 'Population']

df_zscore = data[features_of_interest].apply(zscore)

# %%
outliers = df_zscore.apply(lambda x: (x < -3) | (x > 3))
outlier_indices = outliers.any(axis=1)

# %%
data_cleaned = data[~outlier_indices]

# %%
df_cleaned = data_cleaned.drop(columns=['AveRooms'])

# %%
target = df_cleaned['MedHouseVal']

# %%
df_cleaned = df_cleaned.drop(columns=['MedHouseVal'])

# %%
columns_drop = ['Longitude', 'Latitude','MedHouseVal']

df_cleaned = df_cleaned.drop(columns=columns_drop)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_cleaned, 
                                                    target, 
                                                    test_size=0.2, 
                                                    random_state=42)

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().set_output(transform='pandas').fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
from sklearn.linear_model import LinearRegression
import pandas as pd 

model = LinearRegression().fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

ymin, ymax = y_train.agg(['min', 'max']).values

y_pred = pd.Series(y_pred, index=X_test_scaled.index).clip(ymin, ymax)

print(y_pred.head())

# %%
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

r_sq = model.score(X_train_scaled, y_train)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'R2: {r_sq:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}')

