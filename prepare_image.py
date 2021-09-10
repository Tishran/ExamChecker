########### to do: implement better idea of grade dection and change errors field detection



import cv2
import numpy as np
import os
import re
import torch
import torchvision
import PIL
from read_answers import read_answers
from PIL import Image, ImageFilter
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

CELLS_COORS = ([[(6, 5), (39, 60)],
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
        [(1265, 5), (1299, 60)]],

        [[(3, 0), (36, 60)],
        [(44, 0), (78, 60)], 
        [(87, 0), (120, 60)],
        [(129, 0), (162, 60)],
        [(171, 0), (203, 60)],
        [(213, 0), (246, 60)],
        [(254, 0), (288, 60)],
        [(296, 0), (331, 60)], 
        [(339, 0), (373, 60)], 
        [(379, 0), (414, 60)], 
        [(423, 0), (457, 60)], 
        [(464, 0), (497, 60)], 
        [(506, 0), (541, 60)], 
        [(549, 0), (582, 60)], 
        [(591, 0), (624, 60)], 
        [(633, 0), (665, 60)], 
        [(674, 0), (706, 60)]])

def imageprepare(img):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.fromarray(img)
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (0))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    return np.array(newImage)


def cleaning(roi):
    opening = roi.copy()
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 150:
            cv2.drawContours(opening, [c], -1, 0, -1)

    return opening


def preprocess(roi):
    roi = cleaning(roi)

    if np.sum(roi != 0) < 150:
        return []

    roi = imageprepare(roi)

    return transform(roi).numpy()
    #return roi


def split_to_cell(row, num_of_cells=17, text=False):
    list_of_cells = []

    cells_coors = CELLS_COORS[0]
    if num_of_cells == 17:
        row = cv2.resize(row, (707, 60))
        cells_coors = CELLS_COORS[1]
    elif num_of_cells == 3:
        row = cv2.resize(row, (130, 60))
    elif num_of_cells == 20:
        row = cv2.resize(row, (1305, 60))

    for i in range(num_of_cells):
        imgCell = row[cells_coors[i][0][1] : cells_coors[i][1][1], cells_coors[i][0][0] : cells_coors[i][1][0]]
        imgCell = cv2.resize(imgCell, (128, 128))
        
        grayImgCell = cv2.cvtColor(imgCell, cv2.COLOR_BGR2GRAY)
        grayImgCell = cv2.GaussianBlur(grayImgCell, (9, 9), 0)

        ret, grayImgCell = cv2.threshold(grayImgCell, 140, 255, cv2.THRESH_BINARY_INV) ## maybe change
        
        iterr = 2
        if not text:
            iterr = 1
            grayImgCell = cv2.dilate(grayImgCell, None, iterations=iterr)

        margin_top = 5
        margin_left = 10
        grayImgCell = grayImgCell[margin_top:-margin_top, margin_left:-margin_left]

        res = preprocess(grayImgCell)
        if len(res) > 0:
            list_of_cells.append(res)

    return torch.tensor(list_of_cells), text
    #return list_of_cells, text


def segment_roi_task(list_img, path, kp1, des1, roi, orb, h, w):
    segmented_data = dict()

    for i, y in enumerate(list_img):
        key = int(re.findall(r'\d+', y)[0])

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

        task_num = 1
        error_num = 101
        
        for x, r in enumerate(roi):
            imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]

            h2, w2, c2 = imgCrop.shape
            start_point = 0
            if x < 8:
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

                    segmented_data[key][task_num] = split_to_cell(imgRow, text=isText)
                    task_num += 1
            else:
                segmented_data[key][error_num] = dict()

                imgNumOfText = imgScan[r[0][0][0] : r[0][1][0], r[0][0][1] : r[0][1][1]]
                segmented_data[key][error_num]['num_of_task'] = split_to_cell(imgNumOfText)


                imgRow = imgScan[r[1][0][0] : r[1][1][0], r[1][0][1] : r[1][1][1]]
    
                segmented_data[key][error_num]['answer'] = split_to_cell(imgRow)
                error_num += 1

    return segmented_data


def segment_roi_title(list_img, path, kp1, des1, roi, orb, h, w):
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


            segmented_data[key][r[3]] = split_to_cell(imgCrop, num_of_cells=r[4], text=r[2])

    return segmented_data                                                   
                                                                           
                                                                            
def prepare(paths, tp='both'):                                              
                                                                            
    assert tp in ['both', 'title', 'task'], 'Unsupported type: ' + tp       
                                                                            
    roi1 = [[(296, 690), (1600, 742), True, 'surname', 30],                 
            [(294, 756), (1598, 812), True, 'name', 30],                    
            [(296, 824), (1596, 880), True, 'middlename', 30]]              

    roi2 = [[(147, 587), (854, 882), 'text', '1-5'],
           [(147, 912), (852, 1207), 'text', '6-10'],
           [(147, 1242), (852, 1539), 'text', '11-15'],
           [(144, 1569), (852, 1862), 'text', '16-20'],
           [(927, 587), (1634, 882), 'text', '21-25'], 
           [(927, 912), (1634, 1207), 'text', '26-30'],
           [(927, 1244), (1634, 1539), 'text', '31-35'],
           [(927, 1569), (1634, 1862), 'text', '36-40'],]
           #[[(126, 1955), (195, 2002)], [(219, 1953), (851, 2002)], 'err1'],
           #[[(128, 2015), (197, 2062)], [(219, 2015), (851, 2062)], 'err2'],
           #[[(126, 2075), (195, 2122)], [(219, 2075), (851, 2119)], 'err3'],
           #[[(908, 1955), (979, 2002)], [(999, 1955), (1633, 1999)], 'err4'],
           #[[(911, 2015), (979, 2062)], [(1002, 2013), (1635, 2059)], 'err5'],
           #[[(908, 2073), (979, 2119)], [(999, 2073), (1635, 2117)], 'err6']]

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
