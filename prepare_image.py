########### to do: implement better idea of grade dection and change errors field detection



import cv2
import numpy as np
import os
import re
import torch
import torchvision
import PIL
from read_answers import read_answers
from PIL import ImageTk, Image, ImageDraw
import os, re, math, json, shutil, pprint
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import operator
from scipy.ndimage.morphology import binary_dilation
import imutils

transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])

cells_coors = [[(6, 5), (39, 60)],
        [(48, 5), (82, 60)],
        [(92, 5), (127, 60)],
        [(134, 5), (168, 60)],
        [(180, 5), (212, 60)],
        [(222, 5), (257, 60)],
        [(266, 5), (301, 60)],
        [(308, 5), (345, 60)],
        [(351, 5), (387, 60)],
        [(396, 5), (431, 60)],
        [(442, 5), (476, 60)],
        [(484, 5), (519, 60)], 
        [(527, 5), (561, 60)], 
        [(569, 5), (605, 60)], 
        [(614, 5), (649, 60)], 
        [(657, 5), (693, 60)], 
        [(699, 5), (735, 60)], 
        [(743, 5), (776, 60)], 
        [(787, 5), (820, 60)], 
        [(831, 5), (864, 60)], 
        [(873, 5), (907, 60)], 
        [(918, 5), (951, 60)], 
        [(962, 5), (994, 60)], 
        [(1004, 5), (1036, 60)],
        [(1048, 5), (1079, 60)],
        [(1091, 5), (1124, 60)], 
        [(1135, 5), (1167, 60)], 
        [(1180, 5), (1211, 60)], 
        [(1222, 5), (1255, 60)], 
        [(1265, 5), (1299, 60)]]

WIDTH_OF_ROW = 1305 #<===== trouble linked with this variable


def preprocess_for_text(roi):
    roi = cv2.copyMakeBorder(roi, 25, 25, 25, 25, cv2.BORDER_CONSTANT, None, 0)
    roi = cv2.dilate(roi, None, iterations=2)
    roi = cv2.erode(roi, None, iterations=2)
    roi = cv2.resize(roi, (28, 28))
    ##roi = cv2.normalize(roi, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    ##return np.array([roi])
    return roi


def preprocess_for_num(roi):
    roi = cv2.copyMakeBorder(roi, 15, 15, 15, 15, cv2.BORDER_CONSTANT, None, 0)
    ##return transform(cv2.resize(roi, (28, 28))).numpy()
    return cv2.resize(roi, (28, 28))


def split_to_cell(row, num_of_cells=17, text=False):
    global cells_coors

    list_of_cells = []

    #h, w, c = row.shape
    #div_w = w // num_of_cells
    #start_point = 0

    row = cv2.resize(row, (WIDTH_OF_ROW, 60))

    for i in range(num_of_cells):
        ##imgCell = row[0 : w, start_point : start_point + div_w]
        imgCell = row[cells_coors[i][0][1] : cells_coors[i][1][1], cells_coors[i][0][0] : cells_coors[i][1][0]]
        #imgCell = cv2.resize(imgCell, (128, 128))
        imgCell = cv2.resize(imgCell, (256, 256))
        
        grayImgCell = cv2.cvtColor(imgCell, cv2.COLOR_BGR2GRAY)
        grayImgCell = cv2.GaussianBlur(grayImgCell, (7, 7), 0)

        ret, grayImgCell = cv2.threshold(grayImgCell, 140, 255, cv2.THRESH_BINARY_INV) ## maybe change
        
        iterr = 1
        if text:
            iterr = 2

        grayImgCell = cv2.dilate(grayImgCell, None, iterations=iterr)

        margin_top = 20
        margin_left = 20
        grayImgCell = grayImgCell[margin_top:-(margin_top-15), margin_left:-margin_left]

        if np.sum(grayImgCell != 0) < 600:
            continue

        if text:
            list_of_cells.append(preprocess_for_text(grayImgCell))
        else:
            list_of_cells.append(preprocess_for_num(grayImgCell))

    ##return torch.tensor(list_of_cells), text
    return list_of_cells, text


def segment_roi_task(list_img, path, kp1, des1, roi, orb, h, w):
    global WIDTH_OF_ROW

    segmented_data = dict()

    for i, y in enumerate(list_img):
        key = int(re.findall(r'\d+', y)[0])
        print(key)

        answers_frame = read_answers()[key]

        img = cv2.imread(path + "/" + y)
        img = cv2.resize(img, (w, h))
        h, w, c = img.shape
        segmented_data[key] = dict()

        kp2, des2 = orb.detectAndCompute(img, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des2, des1)
        matches.sort(key=lambda x: x.distance)

        good_matches = matches[:int(len(matches) * 0.25)]

        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
        imgScan = cv2.warpPerspective(img, M, (w, h))
    
        #cv2.imshow('task', imgScan)

        task_num = 1
        error_num = 101
        
        for x, r in enumerate(roi):
            imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]

            h2, w2, c2 = imgCrop.shape
            start_point = 0
            if x < 8:
                WIDTH_OF_ROW = 707

                div_h = h2 // 5
                for k in range(5):
                    isText = False
                    try:
                        ans = answers_frame.iloc[5*x + k]
                        float(ans)
                    except ValueError:
                        isText = True
                    except IndexError:
                        break
                
                    imgRow = imgScan[r[0][1] + start_point : r[0][1] + start_point + div_h, r[0][0] : r[1][0]]
                    start_point += div_h

                    print(imgRow.shape)

                    segmented_data[key][task_num] = split_to_cell(imgRow, text=isText)
                    task_num += 1
            else:
                WIDTH_OF_ROW = 725

                div_h = h2 // 3
                for k in range(3):
                    imgRow = imgScan[r[0][1] + start_point : r[0][1] + start_point + div_h, r[0][0] : r[1][0]]
                    print(imgRow.shape)

                    start_point += div_h
                    segmented_data[key][error_num] = split_to_cell(imgRow)
                    error_num += 1 ################################################### to do: implement better idea for error fixing rows

    return segmented_data


def segment_roi_title(list_img, path, kp1, des1, roi, orb, h, w):
    global WIDTH_OF_ROW

    segmented_data = dict()

    for i, y in enumerate(list_img):
        key = int(re.findall(r'\d+', y)[0])

        img = cv2.imread(path + "/" + y)
        img = cv2.resize(img, (w, h))
        h, w, c = img.shape
        segmented_data[key] = dict()

        kp2, des2 = orb.detectAndCompute(img, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des2, des1)
        matches.sort(key=lambda x: x.distance)
        
        good_matches = matches[:int(len(matches) * 0.25)]
        
        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
        imgScan = cv2.warpPerspective(img, M, (w, h))
        
        for x, r in enumerate(roi):
            imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]

            if r[3] == 'grade':
                WIDTH_OF_ROW = 130
            else:
                WIDTH_OF_ROW = 1305

            segmented_data[key][r[3]] = split_to_cell(imgCrop, num_of_cells=r[4], text=r[2])
    return segmented_data                                                   
                                                                           
                                                                            
def prepare(paths, tp='both'):                                              
                                                                            
    assert tp in ['both', 'title', 'task'], 'Unsupported type: ' + tp       
                                                                            
    roi1 = [[(864, 238), (992, 296), False, 'grade', 3],                    
            [(296, 690), (1600, 742), True, 'surname', 30],                 
            [(294, 756), (1598, 812), True, 'name', 30],                    
            [(296, 824), (1596, 880), True, 'middlename', 30]]              

    roi2 = [[(147, 587), (854, 882), 'text', '1-5'],
           [(147, 912), (852, 1207), 'text', '6-10'],
           [(147, 1242), (852, 1539), 'text', '11-15'],
           [(144, 1569), (852, 1862), 'text', '16-20'],
           [(927, 587), (1634, 882), 'text', '21-25'], 
           [(927, 912), (1634, 1207), 'text', '26-30'],
           [(927, 1244), (1634, 1539), 'text', '31-35'],
           [(927, 1569), (1634, 1862), 'text', '36-40'],
           [(127, 1954), (852, 2119), 'text', 'err1'],
           [(909, 1954), (1632, 2117), 'err2', 'text']]

    query_image1 = cv2.imread('/home/benq/Документы/Rushan/SchoolProject/queries/query_title.png')
    h1, w1, c1 = query_image1.shape

    query_image2 = cv2.imread('/home/benq/Документы/Rushan/SchoolProject/queries/query_task.png')
    h2, w2, c2 = query_image2.shape

    orb = cv2.ORB_create(5000)

    list_of_titles = []
    list_of_tasks = []

    if tp == 'both':
        names_titles = os.listdir(paths[0])
        names_titles.sort()

        names_tasks = os.listdir(paths[1])
        names_tasks.sort()

        kp1, des1 = orb.detectAndCompute(query_image1, None)
        title_data = segment_roi_title(names_titles, paths[0], kp1, des1, roi1, orb, h1, w1)

        kp2, des2 = orb.detectAndCompute(query_image2, None)
        tasks_data = segment_roi_task(names_tasks, paths[1], kp2, des2, roi2, orb, h1, w1)

        return (title_data, tasks_data)
    elif tp=='title':
        names = os.listdir(paths[0])

        kp1, des1 = orb.detectAndCompute(query_image1, None)
        title_data = segment_roi_title(names, paths[0], kp1, des1, roi1, orb, h1, w1)

        return title_data
    else:
        names = os.listdir(paths[0])

        kp2, des2 = orb.detectAndCompute(query_image2, None)
        tasks_data = segment_roi_task(names, paths[0], kp2, des2, roi2, orb, h2, w2)

        return tasks_data


cv2.waitKey(0)
