import cv2
import numpy as np
import os
import torch
import PIL
from PIL import ImageTk, Image, ImageDraw
import os, re, math, json, shutil, pprint
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import operator
from scipy.ndimage.morphology import binary_dilation
import imutils

def split_to_cell(row, num_of_cells=17, text=False):
    list_of_cells = []

    h, w, c = row.shape
    div_w = w // num_of_cells
    start_point = 0
    for i in range(num_of_cells):
        imgCell = row[0 : w, start_point : start_point + div_w]
        imgCell = cv2.resize(imgCell, (128, 128))
        #pim = Image.fromarray(imgCell.astype('uint8'))
        #pim.show()

        #raise TypeError

        grayImgCell = cv2.cvtColor(imgCell, cv2.COLOR_BGR2GRAY)
        grayImgCell = cv2.GaussianBlur(grayImgCell, (7, 7), 0)
        #pim = Image.fromarray(grayImgCell.astype('uint8'))
        #pim.show()

        ret, grayImgCell = cv2.threshold(grayImgCell, 140, 255, cv2.THRESH_BINARY_INV)
        #PIL_image = Image.fromarray(thresh1.astype('uint8'))
        #PIL_image.show()
        
        if not text:
            grayImgCell = cv2.dilate(grayImgCell, None, iterations=1)
        #pim = Image.fromarray(dilateImg.astype('uint8'))
        #pim.show(str())

        cnts = cv2.findContours(grayImgCell.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]

        original = imgCell.copy()

        for cnt in cnts:
            if (cv2.contourArea(cnt) < 125):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            roi = grayImgCell[y:y+h, x:x+w]
            
            if text:
                roi = cv2.copyMakeBorder(roi, 25, 25, 25, 25, cv2.BORDER_CONSTANT, None, 0)

            #cv2.imshow('yay', roi)
            
            list_of_cells.append(cv2.resize(roi, (28, 28)))

        start_point += div_w + 1

    #return torch.tensor(list_of_cells)
    return list_of_cells


def segment_roi_task(list_img, path, kp1, des1, roi, orb, h, w):
    segmented_data = dict()

    for i, y in enumerate(list_img):
        img = cv2.imread(path + "/" + y)
        img = cv2.resize(img, (w, h))
        h, w, c = img.shape
        segmented_data[y] = dict()

        kp2, des2 = orb.detectAndCompute(img, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des2, des1)
        matches.sort(key=lambda x: x.distance)

        good_matches = matches[:int(len(matches) * 0.25)]

        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
        imgScan = cv2.warpPerspective(img, M, (w, h))

        #cv2.imshow('yay', imgScan)

        task_num = 1
        error_num = 101
        
        for x, r in enumerate(roi):            
            imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]

            h2, w2, c2 = imgCrop.shape
            start_point = 0
            if x < 8:
                div_h = h2 // 5
                for k in range(5):
                    imgRow = imgScan[r[0][1] + start_point : r[0][1] + start_point + div_h, r[0][0] : r[1][0]]
                    start_point += div_h
                    segmented_data[y][task_num] = split_to_cell(imgRow)
                    task_num += 1

                    #cv2.imshow('la' + r[3] + str(x), imgRow)
            else:
                div_h = h2 // 3
                for k in range(3):
                    imgRow = imgScan[r[0][1] + start_point : r[0][1] + start_point + div_h, r[0][0] : r[1][0]]
                    start_point += div_h
                    segmented_data[y][error_num] = split_to_cell(imgRow)
                    error_num += 1 # to do: implement better idea for error fixing rows

    return segmented_data


def segment_roi_title(list_img, path, kp1, des1, roi, orb, h, w):
    segmented_data = dict()

    for i, y in enumerate(list_img):
        img = cv2.imread(path + "/" + y)
        img = cv2.resize(img, (w, h))
        h, w, c = img.shape
        segmented_data[y] = dict()

        kp2, des2 = orb.detectAndCompute(img, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des2, des1)
        matches.sort(key=lambda x: x.distance)
        
        good_matches = matches[:int(len(matches) * 0.25)]
        
        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
        imgScan = cv2.warpPerspective(img, M, (w, h))
        
        #cv2.imshow(y, imgScan)
        
        imgShow = imgScan.copy()
        imgMask = np.zeros_like(imgShow)
        
        for x, r in enumerate(roi):
            imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
            segmented_data[y][r[3]] = split_to_cell(imgCrop, r[4])
    return segmented_data


def prepare(paths, tp='both'):

    assert tp in ['both', 'title', 'task'], 'Unsupported type: ' + tp

    roi1 = [[(864, 238), (992, 296), 'text_num', 'grade', 3],
            [(790, 398), (918, 454), 'text', 'subjectname', 3],
            [(964, 396), (1254, 452), 'text_num', 'date', 6],
            [(296, 690), (1600, 742), 'text', 'surname', 30],
            [(294, 756), (1598, 812), 'text', 'name', 30],
            [(296, 824), (1596, 880), 'text', 'middlename', 30]]

   # roi1 = [[(2560, 733), (2953, 900), 'text', 'grade', 3],
   #     [(2353, 1206), (2733, 1373), 'text', 'subject', 3],
   #     [(2866, 1206), (3740, 1366), 'text', 'date', 6],
   #     [(873, 2086), (4766, 2240), 'text', 'surname', 30],
   #     [(873, 2293), (4766, 2446), 'text', 'name', 30],
   #     [(873, 2493), (4766, 2652), 'text', 'middlename', 30]]

  #  roi1 = [[(5049, 1459), (5759, 1759), 'text', 'grade', 3], 
  #         [(4589, 2409), (5359, 2739), 'text', 'subject', 3],
  #         [(5639, 2399), (7349, 2719), 'text', 'date', 6],
  #         [(1639, 4149), (9399, 4469), 'text', 'surname', 30],
  #         [(1629, 4559), (9399, 4869), 'text', 'name', 30],
  #         [(1629, 4959), (9389, 5279), 'text', 'middlename', 30]]
  
   # roi1 = [[(429, 119), (491, 146), 'text', 'grade', 3],
   #         [(392, 198), (456, 226), 'tet', 'subject', 3],
   #         [(478, 198), (622, 224), 'text', 'date', 6],
   #         [(147, 345), (791, 370), 'text', 'surname', 30],
   #         [(147, 379), (792, 402), 'text', 'name', 30],
   #         [(148, 412), (793, 432), 'text', 'middlename', 30]]

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

   # roi2 = [[(142, 579), (857, 884), 'text', '1-5'],
   #         [(142, 904), (854, 1209), 'text', '6-10'],
   #         [(142, 1234), (854, 1542), 'text', '10-15'],
   #         [(139, 1567), (854, 1864), 'text', '15-20'],
   #         [(922, 582), (1639, 882), 'text', '20-24'],
   #         [(922, 907), (1639, 1207), 'text', '26-30'],
   #         [(922, 1234), (1639, 1537), 'text', '31-36'],
   #         [(922, 1564), (1637, 1865), 'text', '36-40'],
   #         [(122, 1949), (857, 2122), 'text', 'chane_errors1'],
   #         [(902, 1947), (1637, 2117), 'text', 'change_errors2']]

   # roi2 = [[(426, 1760), (2553, 2660), 'text', '1-5'],
   #     [(420, 2726), (2560, 3626), 'text', '6-10'],
   #     [(406, 3733), (2540, 4633), 'text', '11-15'],
   #     [(413, 4700), (2553, 5600), 'text', '16-20'],
   #     [(2753, 1753), (4906, 2660), 'text', '21-25'],
   #     [(2753, 2740), (4893, 3626), 'text', '26-30'],
   #     [(2753, 3740), (4900, 4633), 'text', '31-35'],
   #     [(2753, 4713), (4886, 5600), 'text', '36-40'],
   #     [(360, 5853), (2546, 6386), 'text', 'change_errors1'],
   #     [(2700, 5860), (4886, 6366), 'text', 'change_errors2']]

   # roi2 = [[(760, 3506), (4986, 5293), 'text', '1-5'],
   #        [(746, 5440), (5000, 7253), 'text', '6-10'],
   #        [(706, 7426), (5000, 9226), 'text', '11-15'], 
   #        [(746, 9386), (4986, 11173), 'text', '16-20'],
   #        [(5413, 3506), (9666, 5280), 'text', '21-25'], 
   #        [(5413, 5440), (9653, 7226), 'text', '26-30'], 
   #        [(5400, 7440), (9653, 9213), 'text', '31-35'],
   #        [(5400, 9386), (9653, 11146), 'text', '36-40'],
   #        [(560, 11706), (4986, 12693), 'text', 'err1'], 
   #        [(5293, 11666), (9666, 12680), 'text', 'err2']]
  #  roi2 = [[(65, 292), (418, 445), 'text', '1-5'],
  #         [(66, 454), (417, 602), 'text', '6-10'],
  #         [(65, 621), (417, 767), 'text', '11-15'],
  #         [(65, 783), (417, 929), 'text', '16-20'],
  #         [(453, 291), (805, 438), 'text', '21-25'],
  #         [(453, 454), (805, 601), 'text', '26-30'],
  #         [(453, 618), (805, 766), 'text', '31-35'],
  #         [(453, 782), (805, 927), 'text', '36-40'],
  #         [(59, 978), (417, 1057), 'text', 'err1'], 
  #         [(445, 975), (805, 1056), 'text', 'err2']]


    query_image1 = cv2.imread('/home/benq/Документы/Rushan/SchoolProject/queries/query_title.png')
    h1, w1, c1 = query_image1.shape
    #cv2.imshow('query', query_image1)

    query_image2 = cv2.imread('/home/benq/Документы/Rushan/SchoolProject/queries/query_task.png')
    h2, w2, c2 = query_image2.shape

    orb = cv2.ORB_create(2000)

    list_of_titles = []
    list_of_tasks = []

    if tp == 'both':
        names_titles = os.listdir(paths[0])
        names_tasks = os.listdir(paths[1])

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
