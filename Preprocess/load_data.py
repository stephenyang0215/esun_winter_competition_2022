import pandas as pd 


folder = '../dataset'
#ccba
ccba_public = pd.read_csv(rf'{folder}\public_train_x_ccba_full_hashed.csv')
#cdtx
cdtx_public = pd.read_csv(rf'{folder}\public_train_x_cdtx0001_full_hashed.csv')
#custinfo
custinfo_public = pd.read_csv(rf'{folder}\public_train_x_custinfo_full_hashed.csv')
#dp
dp_public = pd.read_csv(rf'{folder}\public_train_x_dp_full_hashed.csv')
#remit
remit_public = pd.read_csv(rf'{folder}\public_train_x_remit1_full_hashed.csv')
#train_alert_time
train_alert_time_public = pd.read_csv(rf'{folder}\train_x_alert_date.csv')
#predict_alert_time
predict_alert_time_public = pd.read_csv(rf'{folder}\public_x_alert_date.csv')
#y
y_public = pd.read_csv(rf'{folder}\train_y_answer.csv')
#案件名單
doc = pd.read_csv(rf'{folder}\預測的案件名單及提交檔案範例.csv')
answer_public = pd.read_csv(rf'{folder}\24_ESun_public_y_answer.csv')
#ccba
ccba_private = pd.read_csv(rf'{folder}\private_x_ccba_full_hashed.csv')
#cdtx
cdtx_private = pd.read_csv(rf'{folder}\private_x_cdtx0001_full_hashed.csv')
#custinfo
custinfo_private = pd.read_csv(rf'{folder}\private_x_custinfo_full_hashed.csv')
#dp
dp_private = pd.read_csv(rf'{folder}\private_x_dp_full_hashed.csv')
#remit
remit_private = pd.read_csv(rf'{folder}\private_x_remit1_full_hashed.csv')
#train_alert_time
train_alert_time_private = pd.read_csv(rf'{folder}\private_x_alert_date.csv')

custinfo = custinfo_public.append(custinfo_private)
dp = dp_public.append(dp_private)
train_alert_time = train_alert_time_public.append(predict_alert_time_public)
y = y_public.append(answer_public)
predict_alert_time = train_alert_time_private.copy()
y = y[~y.alert_key.isin(predict_alert_time.alert_key)].reset_index(drop=True)
doc_merged = custinfo.merge(doc, on='alert_key', how='right').merge(predict_alert_time, on='alert_key', how='left')
doc_merged = doc_merged[~doc_merged['cust_id'].isnull()]
max_doc = doc_merged.groupby('cust_id')['date'].max().reset_index()
max_doc.rename(columns={'date':'max_date'},inplace=True)
doc_merged = doc_merged.merge(max_doc, on='cust_id', how='left')
prev_list = doc_merged[doc_merged['max_date']>doc_merged['date']]['alert_key'].tolist()
