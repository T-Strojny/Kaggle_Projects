import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def apk(actual, predicted, k=10):
    if not len(actual):
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def model_eval(class_model, to_pred):
    model = Pipeline([('pca', PCA(7, random_state=0)), ('svc', class_model)])
    model.fit(X_train, y_train)
    predictions = model.predict_proba(to_pred)

    sorted_prediction_ids = np.argsort(-predictions, axis=1)
    top_3_prediction_ids = sorted_prediction_ids[:, :3]

    original_shape = top_3_prediction_ids.shape
    top_3_predictions = enc.inverse_transform(top_3_prediction_ids.reshape(-1, 1))
    top_3_predictions = top_3_predictions.reshape(original_shape)
    # print(f"{model} MAP@3:", mapk(y_test.values.reshape(-1, 1), top_3_prediction_ids, k=3))
    for i in range(303):
        top_3_predictions[i] = [f"{top_3_predictions[i][0]} {top_3_predictions[i][1]} {top_3_predictions[i][2]}"]
    print(top_3_predictions)
    return top_3_predictions


train_df = pd.read_csv(r"C:\Users\tymek\OneDrive\Pulpit\Kaggle\BornDisesase\BornTrain.csv")
test_df = pd.read_csv(r"C:\Users\tymek\OneDrive\Pulpit\Kaggle\BornDisesase\BornTest.csv")
ids = test_df.id
test_df.drop('id', axis=1, inplace=True)
enc = OrdinalEncoder()
train_df['prognosis'] = [prognosis.replace(' ', '_') for prognosis in train_df['prognosis']]
train_df['prognosis'] = enc.fit_transform(train_df[['prognosis']])


y_train = train_df['prognosis']
X_train = train_df.drop(['prognosis', 'id'], axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)

# model = RandomForestClassifier(random_state=0)
# model.fit(X_train, y_train)
# predictions = model.predict_proba(X_test)
#
#
# sorted_prediction_ids = np.argsort(-predictions, axis=1)
# top_3_prediction_ids = sorted_prediction_ids[:,:3]
# # Because enc.inverse_transform expects a specific shape (a 2D array with 1 column) we can save the original shape to reshape to after decoding
# original_shape = top_3_prediction_ids.shape
# top_3_predictions = enc.inverse_transform(top_3_prediction_ids.reshape(-1, 1))
# top_3_predictions = top_3_predictions.reshape(original_shape)
# print(mapk(y_test.values.reshape(-1, 1), top_3_prediction_ids, k=3))

# model_eval(DecisionTreeClassifier(random_state=1))
# model_eval(SVC(random_state=1, probability=True, C=15, gamma='scale', tol=0.0001, max_iter=1000), to_pred=te)

# voting_model = VotingClassifier([('rf', RandomForestClassifier(random_state=1)),
#                                  ('svm', SVC(random_state=1, probability=True, C=10, kernel='rbf', gamma='scale')),
#                                  ('mlp', MLPClassifier(random_state=1))], voting='soft')
# voting_model.fit(X_train, y_train)
# predictions = voting_model.predict_proba(X_test)
# sorted_prediction_ids = np.argsort(-predictions, axis=1)
# top_3_prediction_ids = sorted_prediction_ids[:,:3]
# # Because enc.inverse_transform expects a specific shape (a 2D array with 1 column) we can save the original shape to reshape to after decoding
# original_shape = top_3_prediction_ids.shape
# top_3_predictions = enc.inverse_transform(top_3_prediction_ids.reshape(-1, 1))
# top_3_predictions = top_3_predictions.reshape(original_shape)
# print(mapk(y_test.values.reshape(-1, 1), top_3_prediction_ids, k=3))
# params = {
#     'C': [0.1, 1, 10,],
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'degree': [2, 3,],
#     'gamma': ['scale', 'auto', 0.1, 1],
#     'coef0': [-1, 0, 1],
#     'shrinking': [True, False],
#     'tol': [1e-3, 1e-4,],
#
# }

# grid_model = GridSearchCV(SVC(probability=True), params, verbose=2.1)
# grid_model.fit(X_train, y_train)
# print(grid_model.best_estimator_)
# SVC(C=0.1, coef0=1, degree=2, gamma=1, kernel='poly', probability=True)
# 4577 i 4405

# lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=8)
# print(lda)
# svm_ld = make_pipeline(lda, SVC(random_state=1, probability=True, C=15, gamma='scale', tol=0.0001, max_iter=1000)).fit(X_train, y_train)
# predictions = svm_ld.predict_proba(X_test)
# sorted_prediction_ids = np.argsort(-predictions, axis=1)
# top_3_prediction_ids = sorted_prediction_ids[:,:3]
# # Because enc.inverse_transform expects a specific shape (a 2D array with 1 column) we can save the original shape to reshape to after decoding
# original_shape = top_3_prediction_ids.shape
# top_3_predictions = enc.inverse_transform(top_3_prediction_ids.reshape(-1, 1))
# top_3_predictions = top_3_predictions.reshape(original_shape)
# print(mapk(y_test.values.reshape(-1, 1), top_3_prediction_ids, k=3))
ids_df = pd.DataFrame(ids)
print(ids_df)
answers_df = pd.DataFrame(model_eval(SVC(random_state=1, probability=True, C=15, gamma='scale', tol=0.0001, max_iter=1000), to_pred=test_df))
print(answers_df)
final_df = pd.concat([ids_df, answers_df], axis=1)
final_df.rename(columns={0:'prognosis', 1:'shit', 2:'shit1'}, inplace=True)
final_df.drop(columns=['shit', 'shit1'], inplace=True)

print(final_df)
final_df.to_csv(r"C:\Users\tymek\OneDrive\Pulpit\Kaggle\BornDisesase\submission_borne4.csv", index=False)