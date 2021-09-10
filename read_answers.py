import os
import re
import pandas as pd

path = './answersTab'
file_names = os.listdir(path)

def read_answers():
    res = dict()

    for i in file_names:
        res[int(re.findall(r'\d+', i)[0])] = pd.read_excel(path + '/' + i, index_col=None, converters={'Ответ': str})['Ответ']


    return res
