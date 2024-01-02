import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA


def model_eval(the_model):
    model = the_model
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    score = mean_absolute_error(y_true=y_val, y_pred=preds)
    print(f'{model} : {score}')


train_df = pd.read_csv('train.csv')
train_df['is_generated'] = 1
test_df = pd.read_csv('test.csv')
ids = test_df.id
other_df = pd.read_csv('CrabAgePrediction.csv')
other_df['is_generated'] = 0
train_df = pd.concat([train_df, other_df], axis=0)

train_df.drop(columns='id', inplace=True)

encoder = OneHotEncoder()
encoded_df = encoder.fit_transform(train_df[['Sex']].values.reshape(-1, 1))
encoded_df = pd.DataFrame(encoded_df.toarray(), columns=['F_Sex_F', 'I_Sex_I', 'M_Sex_M'])
# encoded_df.reset_index(inplace=True)
# encoded_df.drop(columns='index', inplace=True)
# train_df.reset_index(inplace=True)
train_df = pd.concat([train_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
train_df = train_df.drop(columns='Sex')


train_df.drop(columns=['Diameter'], inplace=True)

X = train_df.drop(columns='Age')
# pca_features = X.select_dtypes(include=['float64']).columns.tolist()
# pca = PCA(n_components=4)
# pca.fit(X[pca_features])
# pca_columns = [f'PCA_{i}' for i in range(4)]
# X[pca_columns] = pca.transform(X[pca_features])
print(X)
corr = X.corr()
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = train_df.Age

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=15)

model_eval(RandomForestRegressor(n_jobs=-1, random_state=15))
model_eval(LGBMRegressor(n_jobs=-1, random_state=15))
model_eval(CatBoostRegressor(verbose=False))
model_eval(XGBRegressor())
# 1, 3919