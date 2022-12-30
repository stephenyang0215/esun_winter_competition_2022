import pandas as pd


def train_alert_process_func(data, custinfo, train_alert_time, predict_alert_time, y):
    alert_data = data[data['cust_id'].isin(
        custinfo[custinfo['alert_key'].isin(
            predict_alert_time['alert_key'].tolist())]['cust_id'].tolist())]
    train_data = data[data['cust_id'].isin(
        custinfo[custinfo['alert_key'].isin(
            train_alert_time['alert_key'].tolist())]['cust_id'].tolist())]
    train_data['y'] = 0
    sar_idx = train_data[train_data['cust_id'].isin(
        custinfo[custinfo['alert_key'].isin(
            y[y['sar_flag'] == 1]['alert_key'].tolist())]['cust_id'].tolist())].index
    train_data.loc[sar_idx, 'y'] = 1
    return train_data, alert_data


def train_prev_d(x, day):
    prev_d = x.groupby('cust_id')['tx_date'].max() - day
    prev_d = prev_d.reset_index()
    prev_d.rename(columns={'tx_date': 'prev_d'}, inplace=True)
    x = x.merge(prev_d, on='cust_id', how='left')
    x.drop(x[x['tx_date'] < x['prev_d']].index, inplace=True)
    x.pop('prev_d')
    return x


def preprocess(data):
    """特徵前處理"""
    dict1 = {}
    idx = 0
    num = 0
    for i in range(0, 426, 1):
        dict1[i] = str(idx)
        num += 1
        if num == 7:
            idx += 1
            num = 0
    data['tx_date_group'] = data.tx_date.map(lambda x: dict1[x])
    data['session_cust_id'] = data.tx_date_group + data.cust_id
    data['date_cust_id'] = data.tx_date.astype(str) + data.cust_id
    data['date_time_cust_id'] = data.tx_date.astype(str) + data.tx_date.astype(str) + data.cust_id
    data['tx_year'] = 2021
    data['tx_date_formal'] = data['tx_date'] + 1
    data.loc[data[data['tx_date_formal'] > 365].index, 'tx_year'] = 2022
    data.loc[data[data['tx_date_formal'] > 365].index, 'tx_date_formal'] = \
    data.loc[data[data['tx_date_formal'] > 365].index, 'tx_date_formal'] - 365
    data["tx_date_formal"] = data["tx_year"] * 1000 + data["tx_date_formal"]
    data['tx_date_formal'] = pd.to_datetime(data['tx_date_formal'], format="%Y%j")
    return data


def dp_feature_func(data):
    session_amt_diff = data.groupby(['session_cust_id', 'debit_credit'])['tx_amt'].sum().reset_index()
    session_amt_diff = pd.pivot_table(session_amt_diff, index='session_cust_id', columns='debit_credit', values='tx_amt')
    session_amt_diff.fillna(1, inplace=True)
    session_amt_diff['session_amt_diff_ratio'] = \
    abs(session_amt_diff['CR'] - session_amt_diff['DB']) / abs(session_amt_diff['CR'] + session_amt_diff['DB'])
    session_amt_diff = session_amt_diff.reset_index()[['session_cust_id','session_amt_diff_ratio']]
    data = data.merge(session_amt_diff, on='session_cust_id', how='left')
    #當日 交易差額比率
    date_amt_diff = data.groupby(['date_cust_id', 'debit_credit'])['tx_amt'].sum().reset_index()
    date_amt_diff = pd.pivot_table(date_amt_diff, index='date_cust_id', columns='debit_credit', values='tx_amt')
    date_amt_diff.fillna(1, inplace=True)
    date_amt_diff['date_amt_diff_ratio'] = \
    abs(date_amt_diff['CR'] - date_amt_diff['DB']) / abs(date_amt_diff['CR'] + date_amt_diff['DB'])
    date_amt_diff = date_amt_diff.reset_index()[['date_cust_id', 'date_amt_diff_ratio']]
    data = data.merge(date_amt_diff, on=['date_cust_id'], how='left')
    #當時 交易差額比率
    date_time_amt_diff = data.groupby(['date_time_cust_id', 'debit_credit'])['tx_amt'].sum().reset_index()
    date_time_amt_diff = pd.pivot_table(date_time_amt_diff, index='date_time_cust_id', columns='debit_credit', values='tx_amt')
    date_time_amt_diff.fillna(1, inplace=True)
    date_time_amt_diff['date_time_amt_diff_ratio'] = \
    abs(date_time_amt_diff['CR'] - date_time_amt_diff['DB']) / abs(date_time_amt_diff['CR'] + date_time_amt_diff['DB'])
    date_time_amt_diff = date_time_amt_diff.reset_index()[['date_time_cust_id','date_time_amt_diff_ratio']]
    data = data.merge(date_time_amt_diff, on=['date_time_cust_id'], how='left')
    #當時交易筆數 tx_cnt_date_time
    tx_cnt_date_time = data.groupby(['cust_id','tx_date','tx_time'])['debit_credit'].count().reset_index()
    tx_cnt_date_time.rename(columns={'debit_credit':'tx_cnt_date_time'}, inplace=True)
    data = data.merge(tx_cnt_date_time, on=['cust_id','tx_date','tx_time'], how='left')
    #當日交易筆數 tx_cnt_date
    tx_cnt_date = data.groupby(['cust_id','tx_date'])['debit_credit'].count().reset_index()
    tx_cnt_date.rename(columns={'debit_credit':'tx_cnt_date'}, inplace=True)
    data = data.merge(tx_cnt_date, on=['cust_id','tx_date'], how='left')
    #當時總分行數 txbranch_day_cnt
    txbranch_day_time_cnt = data.groupby(['cust_id','tx_date','tx_time'])['txbranch'].count().reset_index()
    txbranch_day_time_cnt.rename(columns={'txbranch':'txbranch_day_time_cnt'}, inplace=True)
    data = data.merge(txbranch_day_time_cnt, on=['cust_id','tx_date','tx_time'], how='left')
    #單日總分行數 txbranch_day_cnt
    txbranch_day_cnt = data.groupby(['cust_id','tx_date'])['txbranch'].count().reset_index()
    txbranch_day_cnt.rename(columns={'txbranch':'txbranch_day_cnt'}, inplace=True)
    data = data.merge(txbranch_day_cnt, on=['cust_id','tx_date'], how='left')
    #當日ATM 佔交易數比例
    day_atm_txn_ratio = data.groupby(['cust_id','tx_date'])['ATM'].sum().reset_index()
    day_atm_txn_ratio.rename(columns={'ATM':'day_atm_txn_ratio'}, inplace=True)
    data = data.merge(day_atm_txn_ratio, on=['cust_id','tx_date'], how='left')
    data.day_atm_txn_ratio = data.day_atm_txn_ratio / data.tx_cnt_date
    #當時ATM 佔交易數比例
    day_time_atm_txn_ratio = data.groupby(['cust_id','tx_date','tx_time'])['ATM'].sum().reset_index()
    day_time_atm_txn_ratio.rename(columns={'ATM':'day_time_atm_txn_ratio'}, inplace=True)
    data = data.merge(day_time_atm_txn_ratio, on=['cust_id','tx_date','tx_time'], how='left')
    data.day_time_atm_txn_ratio = data.day_time_atm_txn_ratio / data.tx_cnt_date_time
    #當日跨行 佔交易數比例
    day_cross_bank_ratio = data.groupby(['cust_id','tx_date'])['cross_bank'].sum().reset_index()
    day_cross_bank_ratio.rename(columns={'cross_bank':'day_cross_bank_ratio'}, inplace=True)
    data = data.merge(day_cross_bank_ratio, on=['cust_id','tx_date'], how='left')
    data.day_cross_bank_ratio = data.day_cross_bank_ratio / data.tx_cnt_date
    #當時跨行 佔交易數比例
    day_time_cross_bank_ratio = data.groupby(['cust_id','tx_date','tx_time'])['cross_bank'].sum().reset_index()
    day_time_cross_bank_ratio.rename(columns={'cross_bank':'day_time_cross_bank_ratio'}, inplace=True)
    data = data.merge(day_time_cross_bank_ratio, on=['cust_id','tx_date','tx_time'], how='left')
    data.day_time_cross_bank_ratio = data.day_time_cross_bank_ratio / data.tx_cnt_date_time
    #當session交易筆數 tx_cnt_session
    tx_cnt_session = data.groupby(['session_cust_id'])['debit_credit'].count().reset_index()
    tx_cnt_session.rename(columns={'debit_credit':'tx_cnt_session'}, inplace=True)
    data = data.merge(tx_cnt_session, on=['session_cust_id'], how='left')
    #當session總分行數 txbranch_session_cnt
    txbranch_session_cnt = data.groupby(['session_cust_id'])['txbranch'].count().reset_index()
    txbranch_session_cnt.rename(columns={'txbranch':'txbranch_session_cnt'}, inplace=True)
    data = data.merge(txbranch_session_cnt, on=['session_cust_id'], how='left')
    #當session跨行 佔交易數比例
    session_cross_bank_ratio = data.groupby(['session_cust_id'])['cross_bank'].sum().reset_index()
    session_cross_bank_ratio.rename(columns={'cross_bank':'session_cross_bank_ratio'}, inplace=True)
    data = data.merge(session_cross_bank_ratio, on=['session_cust_id'], how='left')
    data.session_cross_bank_ratio = data.session_cross_bank_ratio / data.tx_cnt_date
    #當sessionATM 佔交易數比例
    session_atm_txn_ratio = data.groupby(['session_cust_id'])['ATM'].sum().reset_index()
    session_atm_txn_ratio.rename(columns={'ATM':'session_atm_txn_ratio'}, inplace=True)
    data = data.merge(session_atm_txn_ratio, on=['session_cust_id'], how='left')
    data.session_atm_txn_ratio = data.session_atm_txn_ratio / data.tx_cnt_date
    #time_diff
    distinct_date_time = data[['cust_id','tx_date','tx_time']].drop_duplicates().sort_values(['cust_id','tx_date','tx_time']).reset_index(drop=True)
    distinct_date_time['date_diff'] = distinct_date_time.groupby('cust_id').apply(lambda x: x['tx_date'] - x['tx_date'].shift(1)).reset_index(drop=True)
    distinct_date_time['time_diff'] = distinct_date_time.groupby('cust_id').apply(lambda x: x['tx_time'] - x['tx_time'].shift(1)).reset_index(drop=True)
    distinct_date_time.fillna(0, inplace=True)
    distinct_date_time['time_diff'] = (distinct_date_time['date_diff']*24) + distinct_date_time['time_diff']
    data = data.merge(distinct_date_time[['cust_id', 'tx_date', 'tx_time', 'time_diff']], on=['cust_id', 'tx_date', 'tx_time'], how='left')
    return data


def prev_7d_feature_func(data):
    #last 7 days processing
    data_distinct = data[['cust_id', 'tx_date_formal','tx_date']].drop_duplicates().reset_index(drop=True)
    cross_bank_sum = data.groupby(['cust_id','tx_date_formal'])['cross_bank'].sum().rename('cross_bank_sum').reset_index()
    prev_7d_data = data_distinct.merge(cross_bank_sum, on=['cust_id','tx_date_formal'], how='left')
    dbcr_amt_sum = data.groupby(['cust_id','tx_date_formal','debit_credit'])['tx_amt'].sum().rename('debit_credit_sum').reset_index()
    dbcr_amt_sum = pd.pivot_table(dbcr_amt_sum, index=['cust_id','tx_date_formal'], columns='debit_credit', values='debit_credit_sum')
    prev_7d_data = prev_7d_data.merge(dbcr_amt_sum, on=['cust_id','tx_date_formal'], how='left')
    tx_date_sum = data.groupby(['cust_id','tx_date_formal'])['tx_amt'].sum().rename('tx_amt_sum')
    prev_7d_data = prev_7d_data.merge(tx_date_sum, on=['cust_id','tx_date_formal'], how='left')
    tx_cnt = data.groupby(['cust_id','tx_date_formal'])['tx_amt'].count().rename('tx_cnt').reset_index()
    prev_7d_data = prev_7d_data.merge(tx_cnt, on=['cust_id','tx_date_formal'], how='left')
    #當日ATM 佔交易數比例
    day_atm_cnt = data.groupby(['cust_id','tx_date_formal'])['ATM'].sum().rename('day_atm_cnt').reset_index()
    prev_7d_data = prev_7d_data.merge(day_atm_cnt, on=['cust_id','tx_date_formal'], how='left')
    txbranch_day_cnt = data.groupby(['cust_id','tx_date_formal'])['txbranch'].count().rename('txbranch_day_cnt').reset_index()
    prev_7d_data = prev_7d_data.merge(txbranch_day_cnt, on=['cust_id','tx_date_formal'], how='left')
    prev_7d_data.fillna(0, inplace=True)
    prev_7d_data = prev_7d_data.sort_values(['cust_id','tx_date_formal'])
    prev_7d_data = prev_7d_data.set_index('tx_date_formal')
    #feature engineering
    prev_7d_cross_bank_sum = prev_7d_data.groupby('cust_id')['cross_bank_sum'].rolling(window='7D').sum().rename('prev_7d_cross_bank_sum').reset_index()
    prev_7d_CR_sum = prev_7d_data.groupby('cust_id')['CR'].rolling(window='7D').sum().rename('prev_7d_CR_sum').reset_index()
    prev_7d_DB_sum = prev_7d_data.groupby('cust_id')['DB'].rolling(window='7D').sum().rename('prev_7d_DB_sum').reset_index()
    prev_7d_tx_cnt = prev_7d_data.groupby('cust_id')['tx_cnt'].rolling(window='7D').sum().rename('prev_7d_tx_cnt').reset_index()
    prev_7d_txbranch_cnt = prev_7d_data.groupby('cust_id')['txbranch_day_cnt'].rolling(window='7D').sum().rename('prev_7d_txbranch_cnt').reset_index()
    prev_7d_atm_cnt = prev_7d_data.groupby('cust_id')['day_atm_cnt'].rolling(window='7D').sum().rename('prev_7d_atm_cnt').reset_index()

    data_distinct = data_distinct.merge(prev_7d_CR_sum, on=['cust_id','tx_date_formal']).merge(prev_7d_DB_sum, on=['cust_id','tx_date_formal'])\
    .merge(prev_7d_tx_cnt, on=['cust_id','tx_date_formal']).merge(prev_7d_txbranch_cnt, on=['cust_id','tx_date_formal']).merge(prev_7d_atm_cnt, on=['cust_id','tx_date_formal'])\
    .merge(prev_7d_cross_bank_sum, on=['cust_id','tx_date_formal'])
    data_distinct['prev_7d_txbranch_ratio'] = data_distinct.prev_7d_txbranch_cnt / data_distinct.prev_7d_tx_cnt
    data_distinct['prev_7d_atm_ratio'] = data_distinct.prev_7d_atm_cnt / data_distinct.prev_7d_tx_cnt
    data_distinct['prev_7d_cross_bank_ratio'] = data_distinct.prev_7d_cross_bank_sum / data_distinct.prev_7d_tx_cnt
    data_distinct['prev7d_amt_diff_ratio'] = \
    abs(data_distinct['prev_7d_CR_sum'] - data_distinct['prev_7d_DB_sum']) / abs(data_distinct['prev_7d_CR_sum'] + data_distinct['prev_7d_DB_sum'])
    data_distinct = data_distinct[['cust_id','tx_date','prev_7d_tx_cnt','prev7d_amt_diff_ratio','prev_7d_txbranch_ratio','prev_7d_atm_cnt','prev_7d_atm_ratio',
                                    'prev_7d_txbranch_cnt','prev_7d_cross_bank_ratio','prev_7d_cross_bank_sum']]
    data = data.merge(data_distinct, on=['cust_id', 'tx_date'], how='left')
    return data


def create_features(custinfo, dp, train_alert_time, predict_alert_time, y , output_dir=None):
    """feature engineering"""
    def _pipeline(data, days=30):
        data = (data
                .pipe(train_prev_d, days)
                .pipe(preprocess)
                .pipe(prev_7d_feature_func)
                .pipe(dp_feature_func))
        return data
    train_dp, alert_dp = train_alert_process_func(dp, custinfo,train_alert_time, predict_alert_time, y)
    train_dp = _pipeline(train_dp)
    alert_dp = _pipeline(alert_dp)
    if output_dir:
        train_dp.to_csv(f'{output_dir}/train_dp.csv', index=False)
        alert_dp.to_csv(f'{output_dir}/alert_dp.csv', index=False)
    return train_dp, alert_dp
