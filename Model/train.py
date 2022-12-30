import math
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import catboost as cb
import ngboost as nb 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from .model_cfg import model_cfg


def model_training_1(data, columns):
    """逐筆交易處理"""
    # 僅以DP製作12個特徵來預測
    Y = data['y']
    X = data[columns[1:-1]]
    model = XGBClassifier(**model_cfg['xgb1'])
    model.fit(X, Y)
    feature_importance = pd.DataFrame({'columns':list(X.columns),'score':model.feature_importances_})
    return X, model, feature_importance


def mapping(proba):
    """彙整model training 1偵測結果
        predict_proba以0.1機率區間為一個欄位判斷分佈
    """
    return math.ceil(proba * 10)
   

def debit_credit_ratio_func(data):
    # DB占比
    debit_credit_ratio = data.groupby(['cust_id'])['debit_credit'].value_counts().rename('debit_credit_ratio').reset_index()
    debit_credit_ratio = debit_credit_ratio.pivot_table(values='debit_credit_ratio', index=['cust_id'], columns='debit_credit')
    debit_credit_ratio.fillna(0, inplace=True)
    debit_credit_ratio['debit_credit_ratio'] = debit_credit_ratio['DB']/debit_credit_ratio.sum(axis=1)
    debit_credit_ratio = debit_credit_ratio.reset_index()[['cust_id','debit_credit_ratio']]
    data = data.merge(debit_credit_ratio, on=['cust_id'], how='left')
    return debit_credit_ratio


def all_txn_cnt(data):
    # 總交易次數
    all_txn_cnt = data.groupby('cust_id')['tx_date'].count().rename('all_txn_cnt').reset_index()
    data = data.merge(all_txn_cnt, on='cust_id', how='left')
    return all_txn_cnt


def result_preprocess_func(data, X, model):
    # 把model1 predict的結果以帳號為y軸、機率分十等分為x軸做pivot，並以此歸戶
    # 舉例：A在等級1為0.3、等級2為0.4、等級3為0.4；則A在dp的交易中有30%為等級1（風險最低）的交易，以此類推。
    # 最後加上 DB占比及總交易次數兩個特徵作為model2的input
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


def model_training_2(result):
    """歸戶判斷是否報SAR"""
    # 分別以六種ensamble算法預測機率
    result_col = list(result.columns)
    result_col.remove('cust_id')
    result_col.remove('y')
    model1 = XGBClassifier(**model_cfg['xgb2'])
    model2 = lgb.LGBMClassifier(**model_cfg['lgbm'])
    model3 = nb.NGBClassifier(**model_cfg['ngb'], 
                              Base=DecisionTreeRegressor(criterion='friedman_mse', max_depth=4))
    model4 = cb.CatBoostClassifier(**model_cfg['cb'])
    model5 = GradientBoostingClassifier(**model_cfg['gb'])
    model6 = HistGradientBoostingClassifier(**model_cfg['hgb'])
    model1.fit(result[result_col], result['y'])
    model2.fit(result[result_col], result['y'])
    model3.fit(result[result_col], result['y'])
    model4.fit(result[result_col], result['y'])
    model5.fit(result[result_col], result['y'])
    model6.fit(result[result_col], result['y'])
    model = [model1, model2, model3, model4, model5, model6]
    return model, result_col


def training(train_dp):
    """訓練model1 & model2"""
    dp_col = ['cust_id','session_amt_diff_ratio',
              'date_time_amt_diff_ratio','tx_cnt_date_time',
              'txbranch_day_time_cnt', 'day_time_atm_txn_ratio',
              'day_time_cross_bank_ratio', 'date_amt_diff_ratio',
              'tx_cnt_date','txbranch_day_cnt','day_atm_txn_ratio',
              'day_cross_bank_ratio', 'time_diff', 'y']
    dp_X, dp_model_1, _ = model_training_1(train_dp, dp_col)
    dp_result = result_preprocess_func(train_dp, dp_X, dp_model_1)
    dp_col.remove('y')
    dp_col.remove('cust_id')
    dp_model_2, dp_result_col = model_training_2(dp_result)
    return dp_model_1, dp_model_2, dp_col, dp_result_col
