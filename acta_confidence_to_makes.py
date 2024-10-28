import pandas as pd
import numpy as np

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# OlympIA script that processes and analyzes basketball game footage by detecting specific 
# moments, such as successful scoring attempts, based on confidence values associated
# with video frames. It reads timing cuts, detects high-confidence frames, 
# and exports the relevant frames for further analysis.
# Designed and produced by the OlympIA team.
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



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


def compute_detections(start_match, half_time_start, half_time_final):
    confidence_detections = pd.read_csv(f'{ROOT_DIRECTORY}/frame_confidence_{match}.csv', header=0, names=['id_frame', 'confidence'])
    cont = 0
    i = start_match
    max_value = 0
    max_frame = 0
    makes = []
    while i < len(confidence_detections):
        if i<half_time_start or i>half_time_final:
            conf = confidence_detections['confidence'][i]
            if conf > 0.5:
                cont = cont+1
                if conf > max_value:
                    max_value = conf 
                    max_frame = i
            else:
                cont = 0
            if cont == 6 and max_value>0.7:
                makes.append(max_frame)
                max_value = 0
                i += 400
        i += 1
    return makes


def acta_confidence_to_makes(m, d):
    global match
    global ROOT_DIRECTORY
    match = m
    ROOT_DIRECTORY = d

    frames = cuts_quarters()
    start_match = frames[0][0]
    half_time_start = frames[1][1]
    half_time_final = frames[2][0]
    detections = compute_detections(start_match, half_time_start, half_time_final)
    df_detections = pd.DataFrame({'id_frames': pd.Series(detections)})
    df_detections.to_csv(f'{ROOT_DIRECTORY}/makes_{match}.csv', index=False)



