from prepare_image import prepare
from number_recognition import get_predictions_num
from read_answers import read_answers
from text_recognition import get_predictions_text
import pandas
import cv2


title_data, tasks_data = prepare(['/home/benq/Документы/Rushan/SchoolProject/blanks/1/300', '/home/benq/Документы/Rushan/SchoolProject/blanks/2/300'])

#print(title_data)
#print(tasks_data)
#for i in range(len(title_data['test_title.png']['name'])):
#    cv2.imshow(str(i), title_data['test_title.png']['name'][i])
#
cv2.waitKey(0)

if tasks_data['test_task.png'][6][1]:
    print(get_predictions_text(tasks_data['test_task.png'][6][0]))
else:
    print(get_predictions_num(tasks_data['test_task.png'][6][0]))

#for i in range(len(title_data['test_title.png']['middlename'][0])):
#    cv2.imwrite('./' + str(i) + '.jpg', title_data['test_title.png']['middlename'][0][i])

#for i in range(len(tasks_data['test_task.png'][6][0])):
#    cv2.imwrite('./' + str(i) + '.jpg', tasks_data['test_task.png'][6][0][i])

#true_answers = read_answers()
