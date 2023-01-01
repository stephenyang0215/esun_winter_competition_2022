import pandas as pd
from Preprocess.preprocess import create_features
from Preprocess.load_data import create_dataset, load_doc
from Model.train import training
from Model.inference import create_final_output


if __name__ == "__main__":
    input_dir = './dataset'
    output_dir = None
    doc = load_doc(input_dir)
    custinfo, dp, train_alert_time, predict_alert_time, y = create_dataset(input_dir)
    train_dp, alert_dp = create_features(custinfo, dp, train_alert_time,
                                         predict_alert_time, y, output_dir)
    dp_model_1, dp_model_2, dp_col, dp_result_col = training(train_dp)
    output = create_final_output(alert_dp, dp_model_1, dp_model_2, dp_col, dp_result_col,
                                 doc, custinfo, predict_alert_time)
    output.to_csv('test2.csv', index=False)
    print("OK")
