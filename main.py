from prepare_image import prepare
from number_recognition import get_predictions_num
from read_answers import read_answers
from text_recognition import get_predictions_text
import pandas as pd
import cv2
import os
import re


title_data, tasks_data = prepare(['/home/benq/Документы/Rushan/SchoolProject/answers/1', '/home/benq/Документы/Rushan/SchoolProject/answers/2'])

result_path = './results/'

true_answers = read_answers()
student_answers = dict()
student_data = dict()


for k in tasks_data.keys():
    student_answers[k] = []
    for l in tasks_data[k].values():
        if len(l[0]):
            if l[1]:
                student_answers[k].append(get_predictions_text(l[0]))
            else:
                student_answers[k].append(str(get_predictions_num(l[0])))


for k in title_data.keys():
    student_data[k] = []
    for l in title_data[k].values():
        if len(l[0]):
            student_data[k].append(get_predictions_text(l[0]))


for i in student_answers.keys():
    data = {'True Answers': true_answers[i], 'Student Answers': student_answers[i]}
    resulting_table = pd.DataFrame(data)
    writer = pd.ExcelWriter(result_path + student_data[i][0] + ' ' + student_data[i][1] + ' ' + student_data[i][2] + '.xlsx', engine='xlsxwriter')
    resulting_table.to_excel(writer, index=False, sheet_name='Results')
    
    workbook = writer.book
    worksheet = writer.sheets['Results']

    format_red = workbook.add_format({'bg_color': '#FF0000'})
    format_green = workbook.add_format({'bg_color': "#00ff00"})

    start_row = 1
    start_col = 1
    end_row = len(resulting_table)
    end_col = start_col

    worksheet.conditional_format(start_row, start_col, end_row, end_col, {'type': 'formula', 'criteria': '=$A2<>B2', 'format': format_red})
    worksheet.conditional_format(start_row, start_col, end_row, end_col, {'type': 'formula', 'criteria': '=$A2=B2', 'format': format_green})

    writer.save()    

#list_ofkeys = os.listdir('/home/benq/Документы/Rushan/SchoolProject/blanks/2/300')
#
#for i in title_data[3].keys(): 
##for i in re.findall(r'\d+', list_ofkeys[0]):
#    if title_data[3][i][1] and len(title_data[3][i][0]) > 0:
#        print(get_predictions_text(title_data[3][i][0]))
#    elif len(title_data[3][i][0]) > 0:
#        print(get_predictions_num(title_data[3][i][0]))
#    #    for j in range(len(tasks_data[2][i][0])):
#    #        cv2.imwrite('./Data/' + str(i) + str(j) + ".jpg", tasks_data[2][i][0][j]) ## were str(k) + str(j)

cv2.waitKey(0)
