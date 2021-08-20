import os
import pandas as pd

path = './answersTab'
file_name = os.listdir(path)[0]

def read_answers():
    res = pd.read_excel(path + '/' + file_name, index_col=None)

    print(res)

    return res
