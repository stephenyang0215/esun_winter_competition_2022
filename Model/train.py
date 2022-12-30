import pandas as pd 
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import catboost as cb
import ngboost as nb 
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from Preprocess.preprocess import train_dp, alert_dp


def model_training_1(data, columns):
  #逐筆交易處理
  Y = data['y']
  X = data[columns[1:-1]]
  test_size = 0.2
  seed = 42
  model = XGBClassifier(
                    base_score= 0.5, 
                booster= 'gbtree', 
                colsample_bylevel= 1, 
                colsample_bynode= 1, 
                colsample_bytree= 1, 
                gamma= 0, 
                learning_rate= 0.03,
                max_delta_step= 0, 
                max_depth= 2, 
                min_child_weight= 1, 
#                 missing= None, 
                n_estimators= 90, 
                nthread= 16, 
                objective= 'binary:logistic', 
                reg_alpha= 0, 
                reg_lambda= 1, 
                scale_pos_weight= 1, 
                seed= 0, 
                subsample= 0.9,
                verbosity= 1
  )
  model.fit(X, Y)
  feature_importance = pd.DataFrame({'columns':list(X.columns),'score':model.feature_importances_})
  return X, model, feature_importance

#彙整model training 1偵測結果
#predict_proba以0.05機率區間為一個欄位判斷分佈
def mapping(proba):
    if proba <= 0.1:
        return 1
    elif 0.1 < proba <= 0.2:
        return 2
    elif 0.2 < proba <= 0.3:
        return 3
    elif 0.3 < proba <= 0.4:
        return 4
    elif 0.4 < proba <= 0.5:
        return 5
    elif 0.5 < proba <= 0.6:
        return 6
    elif 0.6 < proba <= 0.7:
        return 7
    elif 0.7 < proba <= 0.8:
        return 8
    elif 0.8 < proba <= 0.9:
        return 9
    elif 0.9 < proba <= 1:
        return 10
def debit_credit_ratio_func(data):
  #id
  debit_credit_ratio = data.groupby(['cust_id'])['debit_credit'].value_counts().rename('debit_credit_ratio').reset_index()
  debit_credit_ratio = debit_credit_ratio.pivot_table(values='debit_credit_ratio', index=['cust_id'], columns='debit_credit')
  debit_credit_ratio.fillna(0, inplace=True)
  debit_credit_ratio['debit_credit_ratio'] = debit_credit_ratio['DB']/debit_credit_ratio.sum(axis=1)
  debit_credit_ratio = debit_credit_ratio.reset_index()[['cust_id','debit_credit_ratio']]
  data = data.merge(debit_credit_ratio, on=['cust_id'], how='left')
  return debit_credit_ratio
def all_txn_cnt(data):
  all_txn_cnt = data.groupby('cust_id')['tx_date'].count().rename('all_txn_cnt').reset_index()
  data = data.merge(all_txn_cnt, on='cust_id', how='left')
  return all_txn_cnt
def result_preprocess_func(data, X, model):
  data_db_cr_ratio = debit_credit_ratio_func(data)
  data_all_txn_cnt = all_txn_cnt(data)
  data['proba'] = model.predict_proba(X)[:,1]
  result = data[['cust_id','proba','y']]
  result['level'] = result["proba"].map(mapping)
  result = result[['cust_id', 'level', 'y']]
  result = result.groupby(['cust_id','level']).count().reset_index().pivot_table(index='cust_id', columns='level', values='y')
  result.fillna(0, inplace=True)
  result = result.div(result.sum(axis=1), axis=0).reset_index()
  for col in list(set([num for num in range(1,11,1)]) -  set(result.columns[1:])):
    result[col] = 0.0
  result = result[['cust_id', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
  result = result.merge(data[['cust_id','y']].drop_duplicates(), on='cust_id', how='left')
  result = result.merge(data_db_cr_ratio, on='cust_id', how='left')
  result = result.merge(data_all_txn_cnt, on='cust_id', how='left')
  return result
#model training 2
#歸戶判斷是否報SAR
def model_training_2(result):
    result_col = list(result.columns)
    result_col.remove('cust_id')
    result_col.remove('y')
    test_size = 0.2
    seed = 42
#     X_train, X_test, y_train, y_test = train_test_split(result[result_col], result['y'], test_size=test_size, random_state=seed)#, stratify = result['y'])
    estimators = [
        ('xgb', model1),
        ('lgbm', model2),
        ('nb', model3),
        ('cat', model4),
        ('gd', model5),
        ('hgbm', model6)
    ]
    
    model1.fit(result[result_col], result['y'])
    model2.fit(result[result_col], result['y'])
    model3.fit(result[result_col], result['y'])
    model4.fit(result[result_col], result['y'])
    model5.fit(result[result_col], result['y'])
    model6.fit(result[result_col], result['y'])
    
    model = [model1, model2, model3, model4, model5, model6]
    
    
    return model, result_col

model1 = XGBClassifier(
                        base_score= 0.5, 
                        booster= 'gbtree', 
                        colsample_bylevel= 1, 
                        colsample_bynode= 1, 
                        colsample_bytree= 1, 
                        gamma= 0, 
                        learning_rate= 0.05,
                        max_delta_step= 0, 
                        max_depth= 3, 
                        min_child_weight= 1, 
                        n_estimators=200, 
                        nthread= 16, 
                        objective= 'binary:logistic', 
                        reg_alpha= 0.1, 
                        reg_lambda= 2, 
                        scale_pos_weight= 1, 
                        seed= 0, 
                        subsample= 1,
                        verbosity= 1
                  )
model2 = lgb.LGBMClassifier(learning_rate = 0.1
                             , max_depth=3
                             , reg_lambda=0
                             , n_estimators=100
                             , reg_alpha=0.01)
model3 = nb.NGBClassifier(
          n_estimators = 500
        , Base = DecisionTreeRegressor(criterion='friedman_mse', max_depth=4)
        , learning_rate = 0.1
  )

model4 = cb.CatBoostClassifier(
                            learning_rate = 0.5
                             , max_depth=3
                             , reg_lambda=2
                             , n_estimators=150
#                              , reg_alpha=0.01
                             , subsample = 1, verbose= 0
  )

model5 = GradientBoostingClassifier(
                learning_rate= 0.2,
                max_depth= 2, 
                n_estimators=100, subsample=1, random_state = 0)

model6 = HistGradientBoostingClassifier(                
              learning_rate= 0.05,
                max_depth= 3, 
#                 missing= None, 
                max_iter=200)




dp_col = ['cust_id','session_amt_diff_ratio', 
          'date_time_amt_diff_ratio','tx_cnt_date_time','txbranch_day_time_cnt', 'day_time_atm_txn_ratio','day_time_cross_bank_ratio',
          'date_amt_diff_ratio','tx_cnt_date','txbranch_day_cnt','day_atm_txn_ratio','day_cross_bank_ratio',#(version : original)
          'time_diff',
          'y']
dp_X, dp_model_1, dp_feature_importance = model_training_1(train_dp, dp_col)
dp_result = result_preprocess_func(train_dp, dp_X, dp_model_1)
dp_col.remove('y')
dp_col.remove('cust_id')
dp_model_2, dp_result_col = model_training_2(dp_result)
