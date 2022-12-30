import pandas as pd


answer = pd.read_csv('test2.csv')
result = pd.read_csv('test.csv')
pd.testing.assert_frame_equal(answer, result)