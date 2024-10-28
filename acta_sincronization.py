# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This script processes and synchronizes timestamp data for basketball game footage,
# particularly aligning scoring events from the game's official record (digital acta)
# with detected moments in the video frames. This alignment ensures accurate timing across 
# the dataset for further analysis or comparison.
# Designed and produced by the OlympIA team.
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def cuts_quarters():
    file = f'{ROOT_DIRECTORY}/cuts_{match}.txt'
    frames = []
    with open(file, "r") as input:
        cont_line = 0
        for line in input:
            if cont_line > 0:
                line = line.replace('\n', '')
                elements = line.split(sep=', ')
                frames.append((int(elements[0]), int(elements[1])))    
            cont_line += 1
    return frames


def get_makes_acta(acta):
    makes2_acta = acta.loc[acta['description']=='canasta de 2']['timestamp']
    makes2b_acta = acta.loc[acta['description']=='mate']['timestamp']
    makes3_acta = acta.loc[acta['description']=='canasta de 3']['timestamp']
    makes_acta = pd.concat([makes2_acta, makes2b_acta, makes3_acta])
    makes_acta = makes_acta.sort_values().tolist()
    return makes_acta


def get_makes_detections():
    detections = pd.read_csv(f'{ROOT_DIRECTORY}/makes_{match}.csv')
    makes_detections = detections['id_frames'].tolist()
    return makes_detections


def gaussian(x, center, width):
    return np.exp(-((x-center) / width)**2 / 2) / np.sqrt(2*np.pi*width**2)


def get_signal_representations(makes_acta, makes_detections):
    centers1 = makes_detections
    centers2 = makes_acta
    start = makes_detections[0]
    stop = makes_acta[-1]
    num_points = stop-start
    width = 3

    x1 = np.linspace(start, stop, num_points)
    y1 = np.zeros(num_points)
    x2 = np.linspace(start, stop, num_points)
    y2 = np.zeros(num_points)

    for center in centers1:
        y1 += gaussian(x1, center, width)
    for center in centers2:
        y2 += gaussian(x2, center, width)
    return y1, y2


def find_offset(v1, v2):
    corr = np.correlate(v1, v2, mode='full')
    max_index = np.argmax(corr)
    offset = len(v2) - 1 - max_index    
    return offset


def apply_offset(x, offset):
    return x - offset


def find_nearest_detection(make, makes_detections, threshold):
    nearest_detection = 0
    for detection in makes_detections:
        if abs(make-detection) < make-nearest_detection and abs(make-detection) < threshold:
            nearest_detection = detection
    if nearest_detection == 0:
        return make
    return nearest_detection


def correct_same_detections(list_frames):
    i = 0
    while i < len(list_frames)-1:
        if list_frames[i][1] == list_frames[i+1][1]:
            if abs(list_frames[i][0] - list_frames[i][1]) < abs(list_frames[i+1][0] - list_frames[i+1][1]): list_frames[i+1][1] = list_frames[i+1][0]
            else: 
                list_frames[i][1] = list_frames[i][0]
        i += 1
    return list_frames


def relate_real_detected_makes(acta, makes_acta, makes_detections):
    timestamps = []
    for make in makes_acta:
        timestamps.append(find_nearest_detection(make, makes_detections, 250))

    frames = cuts_quarters()
    end_halftime_frame = frames[2][0]

    i = 0
    list_frames1 = []
    list_idx1 = []
    list_frames2 = []
    list_idx2 = []
    for index, row in acta.iterrows():
        if row['description'] == 'mate' or row['description'] == 'canasta de 2' or row['description'] == 'canasta de 3':
            if row['timestamp'] < end_halftime_frame:
                list_frames1.append([row['timestamp'], timestamps[i]])
                list_idx1.append(index)
            else: 
                list_frames2.append([row['timestamp'], timestamps[i]])
                list_idx2.append(index)
            acta.at[index, 'timestamp'] = timestamps[i]
            i += 1
    
    list_frames1 = correct_same_detections(list_frames1)
    list_frames2 = correct_same_detections(list_frames2)

    list_idx_act1 = []
    for i in range(len(list_idx1)):
        if list_frames1[i][0]!= list_frames1[i][1]:
            list_idx_act1.append(list_idx1[i])

    list_idx_act2 = []
    for i in range(len(list_idx2)):
        if list_frames2[i][0]!= list_frames2[i][1]:
            list_idx_act2.append(list_idx2[i])

    return acta, list_frames1, list_idx_act1, list_frames2, list_idx_act2


def apply_trans(acta, ctt, a, b, dist):
    idx1 = min(a, b) + 1
    idx2 = max(a, b) - 1
    i = idx1
    while i <= idx2:
        acta.at[i, 'timestamp'] = (acta['timestamp'][i] + dist) * round(ctt)
        i += 1
    return acta 


def apply_transformation(acta, list_frames1, list_idx_act1, list_frames2, list_idx_act2):
    i = 0
    a = 0
    b = 0
    while i < (len(list_idx_act1)-1):
        a = list_idx_act1[i]
        b = list_idx_act1[i+1]
        dif_acta = list_frames1[i+1][0] - list_frames1[i][0]
        dif_video = list_frames1[i+1][1] - list_frames1[i][1]
        dif_ini = list_frames1[i][1] - list_frames1[i][0]
        acta = apply_trans(acta, dif_video/dif_acta, a, b, dif_ini)
        i += 1

    i = 0
    a = 0
    b = 0
    while i < (len(list_idx_act2)-1):
        a = list_idx_act2[i]
        b = list_idx_act2[i+1]
        dif_acta = list_frames2[i+1][0] - list_frames2[i][0]
        dif_video = list_frames2[i+1][1] - list_frames2[i][1]
        dif_ini = list_frames2[i][1] - list_frames2[i][0]
        acta = apply_trans(acta, dif_video/dif_acta, a, b, dif_ini)
        i += 1


    dif_ini1a = 0
    i = 0 
    while dif_ini1a == 0:
        dif_ini1a = list_frames1[i][1] - list_frames1[i][0]
        i += 1
    i = 0
    while i < list_idx_act1[0]:
        acta.at[i, 'timestamp'] = acta['timestamp'][i] + dif_ini1a
        i += 1

    frames = cuts_quarters()
    end_halftime_frame = frames[2][0]
    dif_ini1b = 0
    i = 0 
    while dif_ini1b == 0:
        dif_ini1b = list_frames2[i][1] - list_frames2[i][0]
        i += 1
    i = (acta['timestamp'] >= end_halftime_frame).idxmax()
    while i < list_idx_act2[0]:
        acta.at[i, 'timestamp'] = acta['timestamp'][i] + dif_ini1b
        i += 1

    dif_ini1b = 0
    i = 0 
    while dif_ini1b == 0:
        dif_ini1b = list_frames2[i][1] - list_frames2[i][0]
        i += 1
    i = (acta['timestamp'] >= end_halftime_frame).idxmax()
    while i < list_idx_act2[0]:
        acta.at[i, 'timestamp'] = acta['timestamp'][i] + dif_ini1b
        i += 1
    
    return acta
    

def acta_sincronization(m, d):
    global match
    global ROOT_DIRECTORY
    match = m
    ROOT_DIRECTORY = d

    acta = pd.read_csv(f'{ROOT_DIRECTORY}/events_{match}.csv')

    # Obtain the list of real and detected makes
    makes_acta = get_makes_acta(acta)
    makes_detections = get_makes_detections()

    # Apply constant offset
    y1, y2 = get_signal_representations(makes_acta, makes_detections)
    offset = find_offset(y1, y2)
    #acta = apply_offset(acta, offset)
    acta['timestamp'] = acta['timestamp'].apply(apply_offset, args=[offset])

    # Update the makes in acta frames
    makes_acta = get_makes_acta(acta)

    # Do the final transformation
    acta, list_frames1, list_idx_act1, list_frames2, list_idx_act2 = relate_real_detected_makes(acta, makes_acta, makes_detections)
    acta = apply_transformation(acta, list_frames1, list_idx_act1, list_frames2, list_idx_act2)

    # Save the results
    acta.to_csv(f'{ROOT_DIRECTORY}/offset_acta_{match}vdef.csv', index=False)


