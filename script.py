import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

train_df = pd.read_parquet("train.parquet")
test_df = pd.read_parquet("test.parquet")

print(train_df.head())
print(train_df.info())
print(train_df.describe())

sns.countplot(x='label', data=train_df)
plt.show()

def generate_features(df):
    features = []
    for index, row in df.iterrows():
        dates = row['dates']
        values = row['values']
        data = pd.DataFrame({'value': values}, index=pd.to_datetime(dates))

        for lag in range(1, 7):
            data[f'lag_{lag}'] = data['value'].shift(lag)

        data['rolling_mean'] = data['value'].rolling(window=3).mean()
        data['rolling_std'] = data['value'].rolling(window=3).std()
        data['month'] = data.index.month
        data['day_of_week'] = data.index.dayofweek
        if len(data) > 1:
            data['autocorr'] = acf(data['value'], nlags=1)[1]
        else:
            data['autocorr'] = 0
        data = data.dropna()
        features.append(data.drop(columns=['value']).mean(axis=0).values)
    return pd.DataFrame(features, columns=[f'lag_{i}' for i in range(1, 7)] + ['rolling_mean', 'rolling_std', 'month', 'day_of_week', 'autocorr'])

train_features = generate_features(train_df)
test_features = generate_features(test_df)

train_features.fillna(0, inplace=True)
test_features.fillna(0, inplace=True)

X = train_features
y = train_df['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = HistGradientBoostingClassifier()
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]
print(f'ROC AUC: {roc_auc_score(y_val, y_pred)}')

joblib.dump(model, 'model.pkl')

test_pred = model.predict_proba(test_features)[:, 1]

submission = pd.DataFrame({
    'id': test_df['id'],
    'score': test_pred
})
submission.to_csv('submission.csv', index=False)
def predict(test_df, model_path='model.pkl'):
    model = joblib.load(model_path)
    test_features = generate_features(test_df)
    test_features.fillna(0, inplace=True)
    test_pred = model.predict_proba(test_features)[:, 1]
    submission = pd.DataFrame({
        'id': test_df['id'],
        'score': test_pred
    })
    submission.to_csv('submission.csv', index=False)
predict(test_df)
