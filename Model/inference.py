from Model.train import mapping
import pandas as pd

def all_txn_cnt(data):
  all_txn_cnt = data.groupby('cust_id')['tx_date'].count().rename('all_txn_cnt').reset_index()
  data = data.merge(all_txn_cnt, on='cust_id', how='left')
  return all_txn_cnt


def debit_credit_ratio_func(data):
  #id
  debit_credit_ratio = data.groupby(['cust_id'])['debit_credit'].value_counts().rename('debit_credit_ratio').reset_index()
  debit_credit_ratio = debit_credit_ratio.pivot_table(values='debit_credit_ratio', index=['cust_id'], columns='debit_credit')
  debit_credit_ratio.fillna(0, inplace=True)
  debit_credit_ratio['debit_credit_ratio'] = debit_credit_ratio['DB']/debit_credit_ratio.sum(axis=1)
  debit_credit_ratio = debit_credit_ratio.reset_index()[['cust_id','debit_credit_ratio']]
  data = data.merge(debit_credit_ratio, on=['cust_id'], how='left')
  return debit_credit_ratio


def alert_key_fill_value(dp_final, custinfo, predict_alert_time, value = 0.1):
  test = dp_final.merge(custinfo.merge(predict_alert_time)[['alert_key','cust_id','date']], how='left').sort_values(['cust_id','date'],ascending=False)
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


def create_final_output(alert_dp, dp_model_1, dp_model_2, dp_col, dp_result_col, doc, custinfo, predict_alert_time):
  dp_final, _ = alert_output(alert_dp, dp_model_1, dp_model_2, dp_col, dp_result_col, doc, predict_alert_time, custinfo)
  dp_final['probability'] = dp_final[['probability1', 'probability2', 'probability4', 'probability6']].min(axis=1)
  dp_final2 = alert_key_fill_value(dp_final, custinfo, predict_alert_time, value = 0.1)
  final = doc[['alert_key']].merge(dp_final2, on='alert_key', how='left')
  final = final.fillna(0)
  return final