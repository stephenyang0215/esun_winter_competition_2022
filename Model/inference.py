from Model.train import mapping, all_txn_cnt, debit_credit_ratio_func
import pandas as pd


def alert_key_fill_value(dp_final, custinfo, predict_alert_time, value=0.1):
    """ 補值
    以最後一個date預測機率，前一個則乘上value=0.1的倍數，前兩個則乘上0.01，以此類推

    Args:
        dp_final
        custinfo
        predict_alert_time
        value
    Returns:
        dp_final
    """
    test = dp_final.merge(custinfo.merge(predict_alert_time)[['alert_key', 'cust_id', 'date']], how='left').sort_values(['cust_id','date'],ascending=False)
    test = test[~test['cust_id'].isnull()]
    test_level = test.merge(test[['cust_id','date']].drop_duplicates().groupby('cust_id').apply(lambda x: x['date'].reset_index(drop=True)).reset_index(), how='left')
    test_max = test.merge(test.groupby(['cust_id'])['date'].max().rename('test_max').reset_index().drop_duplicates()).drop_duplicates()
    idx = (test_max['date'] == test_max['test_max'])
    test_max = test_max.loc[idx[idx].index][['cust_id','probability']].rename(columns={'probability':'max_probability'})
    dp_final = test_level.merge(test_max.drop_duplicates())
    dp_final['probability'] = dp_final['probability']*(value**dp_final['level_1'])
    dp_final = dp_final[['alert_key', 'probability']]
    return dp_final


def alert_output(alert, model_1, model_2, alert_col, result_col, doc, predict_alert_time, custinfo):
    """輸出預測結果

    Args:
        alert
        model_1
        model_2
        alert_col
        result_col
        doc
        predict_alert_time
        custinfo
    Returns:
        final_answer
        alert_result
    """
    data_db_cr_ratio = debit_credit_ratio_func(alert)
    data_all_txn_cnt = all_txn_cnt(alert)
    #Remit Submission
    alert_data = alert
    alert_data['proba'] = model_1.predict_proba(alert[alert_col])[:,1]
    alert_result = alert_data[['cust_id','proba']]
    alert_result['level'] = alert_result["proba"].map(mapping)
    alert_result = alert_result[['cust_id', 'level']]
    alert_result['cnt'] = 1
    alert_result = alert_result.groupby(['cust_id','level'])['cnt'].count().reset_index().pivot_table(index='cust_id', columns='level', values='cnt')
    alert_result.fillna(0, inplace=True)
    alert_result = alert_result.div(alert_result.sum(axis=1), axis=0)
    alert_result = alert_result.reset_index()
    for col in list(set([num for num in range(1,11,1)]) -  set(alert_result.columns[1:])):
        alert_result[col] = 0.0
    alert_result = alert_result.merge(data_db_cr_ratio, on='cust_id', how='left')
    alert_result = alert_result.merge(data_all_txn_cnt, on='cust_id', how='left')
    result_col = [str(i) for i in result_col]
    alert_result.columns = [str(i) for i in alert_result.columns]
    final_answer = pd.DataFrame(doc[['alert_key']])
    # predict 6 models probability
    for model in model_2:
        alert_pred = model.predict_proba(alert_result[result_col])
        # evaluate predictions
        alert_result_cust_id = alert_result[['cust_id']]
        alert_result_cust_id['probability'] = alert_pred[:,1]
        final = predict_alert_time.merge(custinfo[['alert_key', 'cust_id']].merge(alert_result_cust_id, on='cust_id'), on='alert_key', how='left')[['alert_key', 'probability']]
        doc = doc[['alert_key']]
        final = doc.merge(final, on='alert_key', how='left')
        final_answer = final_answer.merge(final, on='alert_key', how='left')
    final_answer.columns = ['alert_key', 'probability1', 'probability2', 'probability3', 'probability4', 'probability5', 'probability6']
    return final_answer, alert_result


def create_final_output(alert_dp, dp_model_1, dp_model_2, dp_col, dp_result_col,
                        doc, custinfo, predict_alert_time):
    """create submission file
    依據實踐結果，用最保守的作法輸出，即取其中四個model的預測值取min

    Args:
        alert_dp
        dp_model_1
        dp_model_2
        dp_col
        dp_result_col
        doc
        custinfo
        predict_alert_time
    Returns:
        final
    """
    dp_final, _ = alert_output(alert_dp, dp_model_1, dp_model_2, dp_col, dp_result_col,
                               doc, predict_alert_time, custinfo)
    dp_final['probability'] = dp_final[['probability1', 'probability2',
                                        'probability4', 'probability6']].min(axis=1)
    dp_final2 = alert_key_fill_value(dp_final, custinfo, predict_alert_time, value=0.1)
    final = doc[['alert_key']].merge(dp_final2, on='alert_key', how='left')
    final = final.fillna(0)
    return final
