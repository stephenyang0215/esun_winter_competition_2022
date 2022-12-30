from Model import train, inference

# def train(dataset, config_dict) -> model.pkl
# def inference(dataset, model) -> result.csv
# 
output = inference.final
output.to_csv('test.csv')