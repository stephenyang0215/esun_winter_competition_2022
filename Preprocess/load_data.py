import pandas as pd 


def load_trainset(folder):
    """load trainset (Public)"""
    ccba_public = pd.read_csv(rf'{folder}\public_train_x_ccba_full_hashed.csv')
    cdtx_public = pd.read_csv(rf'{folder}\public_train_x_cdtx0001_full_hashed.csv')
    custinfo_public = pd.read_csv(rf'{folder}\public_train_x_custinfo_full_hashed.csv')
    dp_public = pd.read_csv(rf'{folder}\public_train_x_dp_full_hashed.csv')
    remit_public = pd.read_csv(rf'{folder}\public_train_x_remit1_full_hashed.csv')
    train_alert_time_public = pd.read_csv(rf'{folder}\train_x_alert_date.csv')
    predict_alert_time_public = pd.read_csv(rf'{folder}\public_x_alert_date.csv')
    y_public = pd.read_csv(rf'{folder}\train_y_answer.csv')
    answer_public = pd.read_csv(rf'{folder}\24_ESun_public_y_answer.csv')
    return ccba_public, cdtx_public, custinfo_public, dp_public, remit_public, train_alert_time_public, predict_alert_time_public, y_public, answer_public


def load_testset(folder):
    """load testset (Privite)"""
    ccba_private = pd.read_csv(rf'{folder}\private_x_ccba_full_hashed.csv')
    cdtx_private = pd.read_csv(rf'{folder}\private_x_cdtx0001_full_hashed.csv')
    custinfo_private = pd.read_csv(rf'{folder}\private_x_custinfo_full_hashed.csv')
    dp_private = pd.read_csv(rf'{folder}\private_x_dp_full_hashed.csv')
    remit_private = pd.read_csv(rf'{folder}\private_x_remit1_full_hashed.csv')
    train_alert_time_private = pd.read_csv(rf'{folder}\private_x_alert_date.csv')
    return ccba_private, cdtx_private, custinfo_private, dp_private, remit_private, train_alert_time_private


def load_doc(folder):
    """load output example"""
    doc = pd.read_csv(rf'{folder}\預測的案件名單及提交檔案範例.csv')
    return doc


def create_dataset(folder):
    """concat trainset and testset"""
    (_, _, custinfo_public, dp_public, _, train_alert_time_public,
     predict_alert_time_public, y_public, answer_public) = load_trainset(folder)
    _, _, custinfo_private, dp_private, _, train_alert_time_private = load_testset(folder)
    custinfo = custinfo_public.append(custinfo_private)
    dp = dp_public.append(dp_private)
    train_alert_time = train_alert_time_public.append(predict_alert_time_public)
    y = y_public.append(answer_public)
    predict_alert_time = train_alert_time_private.copy()
    y = y[~y.alert_key.isin(predict_alert_time.alert_key)].reset_index(drop=True)
    return custinfo, dp, train_alert_time, predict_alert_time, y

