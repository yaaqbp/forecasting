import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from math import ceil
import seaborn as sns
import optuna
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error as MSE
import joblib
import warnings
warnings.filterwarnings('ignore')

seed = 123
np.random.seed(seed)



def train_ws_model(n_trials = 10):
    df = pd.read_csv('../data/dummy_weekly_sales.csv')
    X_train, y_train = df[df.year_2012 == 0].drop('Weekly_Sales', axis = 1), df[df.year_2012 == 0][['Weekly_Sales']]
    X_test, y_test = df[df.year_2012 == 1].drop('Weekly_Sales', axis = 1), df[df.year_2012 == 1][['Weekly_Sales']]
    def objective(trial, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test):
        param = {
            'nthread': -1,
            'metric': 'rmse', 
            'random_state': seed,
            'n_estimators': 20000,
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02]),
            'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
            'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
            'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
        }
        model = LGBMRegressor(**param)  
        model.fit(X_train,y_train,eval_set=[(X_test,y_test)],early_stopping_rounds=100,verbose=False)
        preds = model.predict(X_test)
        rmse = MSE(y_test, preds,squared=False)
        return rmse

    study = optuna.create_study(direction='minimize')
    print('weekly sales optuna study created')
    study.optimize(objective, n_trials=n_trials)
    print('weekly sales optuna study optimized')
    params=study.best_params   
    params['random_state'] = seed
    params['n_estimators'] = 20000 
    params['metric'] = 'rmse'
    final_model = LGBMRegressor(**params)
    final_model.fit(X_train,y_train,eval_set=[(X_test,y_test)],early_stopping_rounds=100,verbose=False)
    print('weekly sales model trained')
    joblib.dump(final_model, '../models/lgbm_weekly_sales.pkl')
    print('weekly sales model saved')

def train_ts_model(n_trials = 100):
    df = pd.read_csv('../data/dummy_total_sales.csv')
    X_train, y_train = df[df.year_2012 == 0].drop('total_sales', axis = 1), df[df.year_2012 == 0][['total_sales']]
    X_test, y_test = df[df.year_2012 == 1].drop('total_sales', axis = 1), df[df.year_2012 == 1][['total_sales']]
    def objective(trial, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test):
        param = {
            'n_jobs': -1,
            'metric': 'rmse', 
            'random_state': seed,
            'n_estimators': 20000,
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02]),
            'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
            'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
            'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
        }
        model = LGBMRegressor(**param)  
        model.fit(X_train,y_train,eval_set=[(X_test,y_test)],early_stopping_rounds=100,verbose=False)
        preds = model.predict(X_test)
        rmse = MSE(y_test, preds,squared=False)
        return rmse

    study = optuna.create_study(direction='minimize')
    print('total sales optuna study created')
    study.optimize(objective, n_trials=n_trials)
    print('total sales optuna study optimized')
    params=study.best_params   
    params['random_state'] = seed
    params['n_estimators'] = 20000 
    params['metric'] = 'rmse'
    final_model = LGBMRegressor(**params)
    final_model.fit(X_train,y_train,eval_set=[(X_test,y_test)],early_stopping_rounds=100,verbose=False)
    print('total sales model trained')
    joblib.dump(final_model, '../models/lgbm_total_sales.pkl')
    print('total sales model saved')
"""
def train_markdown_models(n_trials = 10):
    df = pd.read_csv('../data/dummy_markdowns.csv')
    for idx in range(1,6): 
        markdown = f'MarkDown{idx}'
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-5], df[[markdown]], test_size=0.2, random_state=seed)
        def objective(trial, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test):
            param = {
                'n_jobs': -1,
                'metric': 'rmse', 
                'random_state': seed,
                'n_estimators': 20000,
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
                'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
                'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02]),
                'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
                'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
                'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
            }  

            model = LGBMRegressor(**param)  
            model.fit(X_train,y_train,eval_set=[(X_test,y_test)],early_stopping_rounds=100,verbose=False)
            preds = model.predict(X_test)
            rmse = MSE(y_test, preds,squared=False)
            return rmse

        study = optuna.create_study(direction='minimize')
        print(f'markdown{idx} optuna study created')
        study.optimize(objective, n_trials=n_trials)
        print(f'markdown{idx} optuna study optimized')
        params=study.best_params   
        params['random_state'] = seed
        params['n_estimators'] = 20000 
        params['metric'] = 'rmse'
        final_model = LGBMRegressor(**params)
        final_model.fit(X_train,y_train,eval_set=[(X_test,y_test)],early_stopping_rounds=100,verbose=False)
        print(f'markdown{idx} model trained')
        joblib.dump(final_model, f'../models/lgbm_markdown{idx}.pkl')
        print(f'markdown{idx} model saved')
    """

# I have decided not to use optuna in markdowns prediction, as the training time for 5 models was too long.

def train_markdown_models():
    df = pd.read_csv('../data/dummy_markdowns.csv')
    print('markdowns data loaded')
    for idx in range(1,6):
            markdown = f'MarkDown{idx}'
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-5], df[[markdown]], test_size=0.2, random_state=seed)
            iteration = 100000
                                                                                                            
            lgb_params = {                                                                                                                            
                    'nthread': -1,
                    'metric': 'mse',
                    'boosting_type': 'gbdt',    
                    'max_depth': 10,
                    'num_leaves': 40,   
                    'task': 'train',                                                                                                                      
                    'objective': 'regression_l1',                                                                                                         
                    'learning_rate': 0.01,                                                                                                                
                    'feature_fraction': 0.9,                                                                                                              
                    'bagging_fraction': 0.8,                                                                                                              
                    'bagging_freq': 5,                                                                                                                    
                    'lambda_l1': 0.06,                                                                                                                    
                    'lambda_l2': 0.05,                                                                                                                    
                    'verbose': -1,     }                                                                                                                           
                                                                                                                                                    
                                                                                                                            
            lgbtrain_all = lgb.Dataset(data=X_train, label=y_train)                                                       
            final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=iteration)          
            print(f'{markdown} model trained')
            joblib.dump(final_model, f'../models/lgbm_markdown{idx}.pkl') 
            print(f'{markdown} model saved')             


train_ws_model()
train_ts_model()
train_markdown_models()



                               
        