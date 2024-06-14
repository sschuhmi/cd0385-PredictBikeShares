import pandas as pd
from autogluon.tabular import TabularPredictor


train = pd.read_csv("./train.csv")

test = pd.read_csv("./test.csv")

submission = pd.read_csv("./sampleSubmission.csv")

train["datetime"] = pd.to_datetime(train['datetime'])
test["datetime"] = pd.to_datetime(test['datetime'])

train['datetime'].dt.dayofweek

# create new features
train["year"] = train['datetime'].dt.year
train["month"] = train['datetime'].dt.month
train["hour"] = train['datetime'].dt.hour
train["day"] = train['datetime'].dt.dayofweek

test["year"] = test['datetime'].dt.year
test["month"] = test['datetime'].dt.month
test["hour"] = test['datetime'].dt.hour
test["day"] = test['datetime'].dt.dayofweek

train["season"] = train["season"].astype("category")
train["weather"] = train["weather"].astype("category")
test["season"] = test["season"].astype("category")
test["weather"] = test["weather"].astype("category")


remove_columns_list = ['casual', 'registered']
col_names  =[x for x in list(train.columns) if x not in remove_columns_list]

hyperparameters = {
            'GBM': [
                {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
                {},
                'GBMLarge',
            ],
            'CAT': {"learning_rate": 0.03, "iterations": 15, "l2_leaf_reg": 0.125},
            'XGB': {},
            'FASTAI': {},
            'RF': [
                {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
            ],
            'XT': [
                {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
            ]
        }

predictor_new_hpo = TabularPredictor(label="count", problem_type="regression", eval_metric="root_mean_squared_error").fit(train_data=train[col_names], time_limit=1000, presets="best_quality", hyperparameters=hyperparameters)
predictor_new_hpo.fit_summary()
predictions = predictor_new_hpo.predict(test)

submission["count"] = predictions