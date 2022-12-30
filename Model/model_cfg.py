model_cfg = {
    "xgb1": {
        'base_score': 0.5, 
        'booster': 'gbtree', 
        'colsample_bylevel': 1, 
        'colsample_bynode': 1, 
        'colsample_bytree': 1, 
        'gamma': 0, 
        'learning_rate': 0.03,
        'max_delta_step': 0, 
        'max_depth': 2, 
        'min_child_weight': 1, 
        'n_estimators': 90, 
        'nthread': 16, 
        'objective': 'binary:logistic', 
        'reg_alpha': 0, 
        'reg_lambda': 1, 
        'scale_pos_weight': 1, 
        'seed': 0, 
        'subsample': 0.9,
        'verbosity': 1
        },
    "xgb2": {
        'base_score': 0.5, 
        'booster': 'gbtree', 
        'colsample_bylevel': 1, 
        'colsample_bynode': 1, 
        'colsample_bytree': 1, 
        'gamma': 0, 
        'learning_rate': 0.05,
        'max_delta_step': 0, 
        'max_depth': 3, 
        'min_child_weight': 1, 
        'n_estimators': 200, 
        'nthread': 16, 
        'objective': 'binary:logistic', 
        'reg_alpha': 0.1, 
        'reg_lambda': 2, 
        'scale_pos_weight': 1, 
        'seed': 0, 
        'subsample': 1,
        'verbosity': 1
    },
    'lgbm': {
        'learning_rate': 0.1,
        'max_depth': 3,
        'reg_lambda': 0,
        'n_estimators': 100,
        'reg_alpha': 0.01,
    },
    'ngb': {
        'n_estimators': 500,
        'learning_rate': 0.1,
    },
    'cb': {
        'learning_rate': 0.5,
        'max_depth': 3,
        'reg_lambda': 2,
        'n_estimators': 150,
        'subsample': 1,
        'verbose': 0,
    },    
    'gb': {
        'learning_rate': 0.2,
        'max_depth': 2,
        'n_estimators': 100,
        'subsample': 1,
        'random_state': 0,
    },
    'hgb': {
        'learning_rate': 0.05,
        'max_depth': 3,
        'max_iter':200,
    },
}