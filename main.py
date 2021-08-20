from prepare_image import prepare
from number_recognition import get_predictions_num
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

if tasks_data['test_task.png'][8][1]:
    print(get_predictions_text(tasks_data['test_task.png'][8][0]))
else:
    print(get_predictions_num(tasks_data['test_task.png'][8][0]))

#for i in range(len(tasks_data['test_task.png'][8])):
#    cv2.imwrite('./' + str(i) + '.jpg', tasks_data['test_task.png'][8][i])


























#import cv2
#import numpy as np
#import os
#
#def split_to_cell(row, num_of_cells=17):
#    list_of_cells = []
#
#    margin = 25
#    h, w, c = row.shape
#    div_w = w // num_of_cells
#    start_point = 0
#    for i in range(num_of_cells):
#        imgCell = row[0 : w, start_point : start_point + div_w]
#        imgCell = imgCell[margin : -(margin-15), margin : -margin]
#        imgCell = cv2.resize(imgCell, (28, 28))
#        list_of_cells.append(imgCell)
#        start_point += div_w + 1
#
#    return list_of_cells
#
#
#def segment_roi_task(list_img, path, kp1, des1, roi):
#    segmented_data = dict()
#
#    for i, y in enumerate(list_img):
#        img = cv2.imread(path + "/" + y)
#        h, w, c = img.shape
#        segmented_data[y] = dict()
#
#        kp2, des2 = orb.detectAndCompute(img, None)
#        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#        matches = bf.match(des2, des1)
#        matches.sort(key=lambda x: x.distance)
#
#        good_matches = matches[:int(len(matches) * 0.25)]
#
#        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#        dstPoints = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#        
#        M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
#        imgScan = cv2.warpPerspective(img, M, (w, h))
#        
#        imgShow = imgScan.copy()
#        imgMask = np.zeros_like(imgShow)
#
#        task_num = 1
#        error_num = 101
#        
#        for x, r in enumerate(roi):            
#            imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
#
#            h2, w2, c2 = imgCrop.shape
#            start_point = 0
#            if x < 8:
#                div_h = h2 // 5
#                for k in range(5):
#                    imgRow = imgScan[r[0][1] + start_point : r[0][1] + start_point + div_h, r[0][0] : r[1][0]]
#                    start_point += div_h
#                    segmented_data[y][task_num] = split_to_cell(imgRow)
#                    task_num += 1
#
#                    #cv2.imshow('la' + r[3] + str(x), imgRow)
#            else:
#                div_h = h2 // 3
#                for k in range(3):
#                    imgRow = imgScan[r[0][1] + start_point : r[0][1] + start_point + div_h, r[0][0] : r[1][0]]
#                    start_point += div_h
#                    segmented_data[y][error_num] = split_to_cell(imgRow)
#                    error_num += 1
#
#    return segmented_data
#
#def segment_roi_title(list_img, path, kp1, des1, roi):
#    segmented_data = dict()
#
#    for i, y in enumerate(list_img):
#        img = cv2.imread(path + "/" + y)
#        h, w, c = img.shape
#        segmented_data[y] = dict()
#
#        kp2, des2 = orb.detectAndCompute(img, None)
#        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#        matches = bf.match(des2, des1)
#        matches.sort(key=lambda x: x.distance)
#        
#        good_matches = matches[:int(len(matches) * 0.25)]
#        
#        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#        dstPoints = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#        
#        M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
#        imgScan = cv2.warpPerspective(img, M, (w, h))
#        
#        #cv2.imshow(y, imgScan)
#        
#        imgShow = imgScan.copy()
#        imgMask = np.zeros_like(imgShow)
#        
#        for x, r in enumerate(roi1):
#            imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
#            segmented_data[y][r[3]] = split_to_cell(imgCrop, r[4])
#    return segmented_data
#
#
#query_image1 = cv2.imread('/home/benq/Документы/Rushan/SchoolProject/queries/query1m.png')
#height1, width1, c1 = query_image1.shape
#
#query_image2 = cv2.imread('/home/benq/Документы/Rushan/SchoolProject/queries/query2m.png')
#height2, width2, c2 = query_image2.shape
#
#orb = cv2.ORB_create(6000)
#
#path1 = '/home/benq/Документы/Rushan/SchoolProject/blanks/1/'
#blank_list1 = os.listdir(path1)
#
#path2 = '/home/benq/Документы/Rushan/SchoolProject/blanks/2/'
#blank_list2 = os.listdir(path2)
#
#roi1 = [[(2560, 733), (2953, 900), 'text', 'grade', 3],
#        [(2353, 1206), (2733, 1373), 'text', 'subject', 3],
#        [(2866, 1206), (3740, 1366), 'text', 'date', 6],
#        [(873, 2086), (4766, 2240), 'text', 'surname', 30],
#        [(873, 2293), (4766, 2446), 'text', 'name', 30],
#        [(873, 2493), (4766, 2652), 'text', 'middlename', 30]]
#
#roi2 = [[(426, 1760), (2553, 2660), 'text', '1-5'],
#        [(420, 2726), (2560, 3626), 'text', '6-10'],
#        [(406, 3733), (2540, 4633), 'text', '11-15'],
#        [(413, 4700), (2553, 5600), 'text', '16-20'],
#        [(2753, 1753), (4906, 2660), 'text', '21-25'],
#        [(2753, 2740), (4893, 3626), 'text', '26-30'],
#        [(2753, 3740), (4900, 4633), 'text', '31-35'],
#        [(2753, 4713), (4886, 5600), 'text', '36-40'],
#        [(360, 5853), (2546, 6386), 'text', 'change_errors1'],
#        [(2700, 5860), (4886, 6366), 'text', 'change_errors2']]
#
#kp1, des1 = orb.detectAndCompute(query_image1, None)
#title_data = segment_roi_title(blank_list1, path1, kp1, des1, roi1)
#print(title_data)
#
#kp2, des2 = orb.detectAndCompute(query_image2, None)
#tasks_data = segment_roi_task(blank_list2, path2, kp2, des2, roi2)
#print(tasks_data)
#for i in range(len(tasks_data['Test2.png'][6])):
#    print(tasks_data['Test2.png'][1][i].shape)
#    cv2.imshow(str(i), tasks_data['Test2.png'][6][i])
#
#cv2.waitKey(0)
