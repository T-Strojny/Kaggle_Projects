import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import optuna

#
# train_df = pd.read_csv(r"C:\Users\tymek\OneDrive\Pulpit\Kaggle\Regresion Costs\train.csv")
# test_df = pd.read_csv(r"C:\Users\tymek\OneDrive\Pulpit\Kaggle\Regresion Costs\test.csv\test.csv")
# ids = test_df.id
# train_df.drop(columns=['id', 'salad_bar', 'video_store', 'florist'], inplace=True)
# test_df.drop(columns=['id', 'salad_bar', 'video_store', 'florist'], inplace=True)
# y_train = train_df.cost
# X_train = train_df.drop(['cost'], axis=1)
#
# # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
#
#
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_submission_scaled = scaler.transform(test_df)
# X_train_scaled = scaler.transform(X_train)
# print('Hello')
#
# # def objective(trial):
# #     param = {
# #         'max_depth': trial.suggest_int('max_depth', 1, 10),
# #         'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
# #         'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
# #         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
# #         'gamma': trial.suggest_float('gamma', 0.01, 1.0),
# #         'subsample': trial.suggest_float('subsample', 0.01, 1.0),
# #         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
# #         'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
# #         'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
# #         'random_state': trial.suggest_int('random_state', 1, 1000)
# #     }
# #     model = XGBRegressor(**param)
# #     model.fit(X_train, y_train)
# #     y_pred = model.predict(X_test)
# #     return mean_squared_error(y_test, y_pred)
# #
# # study = optuna.create_study(direction='minimize', study_name='regression')
# # study.optimize(objective, n_trials=100)
# # print('Best parameters', study.best_params)
# # print('Best value', study.best_value)
# # print('Best trial', study.best_trial)
# # XGB IS BEST 0.30371
# params = {'max_depth': 10, 'learning_rate': 0.034064854486150736, 'n_estimators': 272, 'min_child_weight': 3, 'gamma': 0.9716541520136038, 'subsample': 0.8205757564609685, 'colsample_bytree': 0.9425793543274713, 'reg_alpha': 0.19694706691272076, 'reg_lambda': 0.6373067571446813, 'random_state': 386}
#
# xgb_model = XGBRegressor(max_depth=10, learning_rate=0.03406, n_estimators=272, min_child_weight=3, gamma=0.9716, subsample=0.8205, colsample_bytree=0.9425, reg_alpha=0.19694, reg_lambda=0.63730)
# xgb_model.fit(X_train_scaled, y_train)
# xgb_model_pred = xgb_model.predict(X_submission_scaled)
# final_df = pd.DataFrame(xgb_model_pred)
# ids_df = pd.DataFrame(ids)
# final_df = pd.concat([ids_df, final_df], axis=1)
# final_df.rename(columns={0:'cost'}, inplace=True)
# final_df.cost = final_df.cost.astype(float)
# final_df.cost.round(3)
# print(final_df.head())
# final_df.to_csv(r"C:\Users\tymek\OneDrive\Pulpit\Kaggle\Regresion Costs\cost_submission.csv", index=False)

test_DF = pd.read_csv(r"C:\Users\tymek\OneDrive\Pulpit\Kaggle\Regresion Costs\cost_submission.csv")
print(test_DF.cost)
test_DF['cost'] = test_DF['cost'].round(decimals=2)
print(test_DF.cost)
test_DF.to_csv(r"C:\Users\tymek\OneDrive\Pulpit\Kaggle\Regresion Costs\cost_submission.csv", index=False)