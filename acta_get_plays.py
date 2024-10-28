# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This OlympIA script is the second flagship product, it produces a csv file
# which contains info about each play of the match. This is done by loading the 
# augmented acta script created in the pipeline. It cuts the match into plays,
# describing each play by the metrics computed in the pipeline. 
# Designed and produced by the OlympIA team.
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt



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


def get_list_detections(file):
    positions = []
    positions_frame = []
    i = 1
    is_first = True
    dic_idx_frame = {}
    idx = 0
    with open(file, "r") as input:
        for line in input:
            line = line.replace('\n', '')
            elements = line.split(sep=' ')
            frame = int(elements[0])
            class_p = int(elements[1])-1
            xpos = float(elements[2])
            ypos = float(elements[3])
            if is_first:
                i = frame
                is_first = False
            if frame != i:
                positions.append([i, positions_frame])
                dic_idx_frame[i] = idx
                idx += 1
                i = frame
                positions_frame = []
            positions_frame.append((class_p, xpos, ypos))
    return positions, dic_idx_frame


def correct_positions(vector):
    # Iterate through the vector
    new_vector = []
    for i in range(len(vector)):
        # Get the current number and its corresponding sublist
        curr_number = vector[i][0]
        sublist = vector[i][1]

        # Check if there is a gap between the current number and the previous number
        if i > 0 and curr_number - vector[i-1][0] > 1:
            # Insert the missing vectors with empty second positions
            for missing_number in range(vector[i-1][0] + 1, curr_number):
                new_vector.append([missing_number, []])

        if len(sublist) < 5:
            sublist = []

        # Add the current vector to the new vector
        new_vector.append([curr_number, sublist])
    return new_vector


def correct_diccionary(original_dict):
    new_dict = {}

    min_key = min(original_dict.keys())
    max_key = max(original_dict.keys())

    # Iterate through the range of numbers from min_key to max_key
    for num in range(min_key, max_key + 1):
        # Check if the number exists in the original dictionary
        if num in original_dict:
            # Add the number and its corresponding value to the new dictionary
            new_dict[num] = original_dict[num]
        else:
            # Find the previous key in the original dictionary
            prev_key = num - 1
            while prev_key not in original_dict:
                prev_key -= 1

            # Increment the value of the previous key and add the number to the new dictionary
            new_dict[num] = original_dict[prev_key] + 1
    return new_dict


def get_detections(positions, dic_idx_frame, frame):
    return positions[dic_idx_frame[frame]][-1]


def add_play(plays, ini, fin, event, team='-', actor='-', timeout=False):
    plays.loc[len(plays)] = pd.Series([[ini, fin], event, team, actor, timeout],
                                      index=["play", "final_event", "team", "actor", "after_timeout"])
    return plays


def get_cuts(acta):
    plays = pd.DataFrame(columns=["play", "final_event", "team", "actor", "after_timeout"])
    plays['after_timeout'] = plays['after_timeout'].astype(bool)
    ini = None
    fin = None
    post_canasta = False
    frame_made = 0
    post_shotmiss = False
    miss_info = [0, 0, 'C']
    timeout = False
    for index, row in acta.iterrows():
        if row['description'] == 'cambio atacante':
            # if not post_clockon:
            if post_shotmiss:
                fin = row['timestamp']
                plays = add_play(plays, ini, fin, f"shot_miss_{int(miss_info[0])}", miss_info[1], int(
                    miss_info[2]), timeout=timeout)
                ini = fin
                fin = None
                post_shotmiss = False
                timeout = False
            elif not post_canasta:
                if ini != None:
                    fin = row['timestamp']
                    plays = add_play(plays, ini, fin, "steal", timeout=timeout)
                    timeout = False
                    ini = fin
                    fin = None
                else:
                    ini = row['timestamp']
            else:
                ini = row['timestamp']
                post_canasta = False

        if row['description'] == 'intento fallado de 2' or row['description'] == 'intento fallado de 3':
            post_canasta = False
            post_shotmiss = True
            miss_info = [row['description'][-1], row['team'], row['actor']]

        if row['description'] == 'clock_on':
            post_canasta = False
            ini = row['timestamp']

        if row['description'] == 'inicio periodo' or row['description'] == 'reinicio despues de falta' or row['description'] == 'reinicio despues de timeout':
            post_canasta = False
            ini = row['timestamp']

        if row['description'] == 'clock_off':
            fin = row['timestamp']
            if post_shotmiss:
                plays = add_play(plays, ini, fin, f"shot_miss_{int(miss_info[0])} + violation", miss_info[1], int(
                    miss_info[2]))
            else:
                plays = add_play(plays, ini, fin, "violation", timeout=timeout)
            post_shotmiss = False
            post_canasta = False
            timeout = False
            ini = None
            fin = None

        if row['description'] == 'tiempo muerto' or row['description'] == 'fin periodo':
            fin = row['timestamp']
            plays = add_play(plays, ini, fin, row['description'], timeout=timeout)
            post_shotmiss = False
            post_canasta = False
            ini = None
            fin = None
            timeout = row['description'] == 'tiempo muerto'

        if row['description'] == 'personal' or row['description'] == 'personal 2tl' or row['description'] == 'personal 3tl' or row['description'] == 'personal 1tl':
            fin = row['timestamp']
            if post_shotmiss:
                plays = add_play(plays, ini, fin, f"shot_miss_{int(miss_info[0])} + foul", miss_info[1], int(
                    miss_info[2]))
            else:
                plays = add_play(plays, ini, fin, "foul",
                                 row['team'], int(row['actor']), timeout=timeout)
            post_shotmiss = False
            post_canasta = False
            timeout = False
            ini = None
            fin = None

        if row['description'] == 'mate' or row['description'] == 'canasta de 2' or row['description'] == 'canasta de 3':
            fin = row['timestamp']
            points = row['description'][-1]
            if points == 'e':
                points = 2
            plays = add_play(plays, ini, fin, f"shot_made_{points}", row['team'], int(
                row['actor']), timeout=timeout)
            post_shotmiss = False
            post_canasta = True
            timeout = False
            frame_made = row['timestamp']
            ini = row['timestamp']
            fin = None

    return plays


def duplicate_last(players):
    players.append(set())
    for element in players[-2]:
        players[-1].add(element)
    return players


def compute_players(acta):
    playersA = [set()]
    playersB = [set()]
    indx_posA = 0
    indx_posB = 0
    indx_posesionsA = {}
    indx_posesionsB = {}
    iniA = 0
    iniB = 0
    is_firstA = True
    is_firstB = True

    for index, row in acta.iterrows():
        p = row['actor']
        t = row['team']
        if t == 'A':
            if pd.notna(p) and p != None:
                p = int(p)
                if row['description'] != 'entra a pista' and row['description'] != 'sale a banquillo':
                    is_firstA = True
                    playersA[-1].add(p)
                else:
                    if is_firstA:
                        is_firstA = False
                        playersA = duplicate_last(playersA)
                        indx_posesionsA[indx_posA] = (iniA, row['timestamp'])
                        indx_posA += 1
                        iniA = row['timestamp']
                    if row['description'] == 'entra a pista':
                        playersA[-1].add(p)
                    if row['description'] == 'sale a banquillo':
                        if p in playersA[-1]:
                            playersA[-1].remove(p)
                        else:
                            playersA[-2].add(p)
        if t == 'B':
            if pd.notna(p) and p != None:
                p = int(p)
                if row['description'] != 'entra a pista' and row['description'] != 'sale a banquillo':
                    is_firstB = True
                    playersB[-1].add(p)
                else:
                    if is_firstB:
                        is_firstB = False
                        playersB = duplicate_last(playersB)
                        indx_posesionsB[indx_posB] = (iniB, row['timestamp'])
                        indx_posB += 1
                        iniB = row['timestamp']
                    if row['description'] == 'entra a pista':
                        playersB[-1].add(p)
                    if row['description'] == 'sale a banquillo':
                        if p in playersB[-1]:
                            playersB[-1].remove(p)
                        else:
                            playersB[-2].add(p)

    playersA = duplicate_last(playersA)
    indx_posesionsA[indx_posA] = (iniA, row['timestamp'])
    playersB = duplicate_last(playersB)
    indx_posesionsB[indx_posB] = (iniB, row['timestamp'])
    return indx_posesionsA, indx_posesionsB, playersA, playersB


def compute_points(acta):
    pointsA = []
    pointsB = []
    indx_posA = 0
    indx_posB = 0
    indx_posesionsA = {}
    indx_posesionsB = {}
    iniA = 0
    iniB = 0
    prevA = 0
    prevB = 0
    for index, row in acta.iterrows():
        A = row['A']
        B = row['B']
        if pd.notna(A) and pd.notna(B) and index > 0:
            if A != prevA:
                pointsA.append(prevA)
                indx_posesionsA[indx_posA] = (iniA, row['timestamp'])
                prevA = A
                iniA = row['timestamp']
                indx_posA += 1
            if B != prevB:
                pointsB.append(prevB)
                indx_posesionsB[indx_posB] = (iniB, row['timestamp'])
                prevB = B
                iniB = row['timestamp']
                indx_posB += 1
    indx_posesionsA[indx_posA] = (iniA, row['timestamp'])
    indx_posesionsB[indx_posB] = (iniB, row['timestamp'])
    pointsA.append(prevA)
    pointsB.append(prevB)
    return indx_posesionsA, indx_posesionsB, pointsA, pointsB


def identify_team(positions, dic_idx_frame, frames, quarter):
    values = []
    for frame in range(frames[0], frames[1]+1):
        for element in get_detections(positions, dic_idx_frame, frame):
            values.append(element[1])
    if len(values) == 0:
        return '-'
    mean = sum(values)/len(values)
    if quarter == '1' or quarter == '2':
        if mean >= 0.5:
            return 'B'
        return 'A'
    else:
        if mean >= 0.5:
            return 'A'
        return 'B'


def compute_info_frame(frame, indx_posesions, info):
    for i in range(len(indx_posesions)):
        if frame >= indx_posesions[i][0] and frame <= indx_posesions[i][1]:
            return info[i]


def add_players(acta, plays):
    indx_posesionsA, indx_posesionsB, playersA, playersB = compute_players(acta)
    plays = plays.assign(playersA='', playersB='')
    for index, row in plays.iterrows():
        plays.loc[index, 'playersA'] = compute_info_frame(
            row['play'][1], indx_posesionsA, playersA)
        plays.loc[index, 'playersB'] = compute_info_frame(
            row['play'][1], indx_posesionsB, playersB)
    return plays


def add_quarters(plays):
    quarters_frames = cuts_quarters()
    plays = plays.assign(quarter='')
    for index, row in plays.iterrows():
        if row['play'][1] < quarters_frames[0][1]:
            row['quarter'] = '1'
        elif row['play'][1] < quarters_frames[1][1]:
            row['quarter'] = '2'
        elif row['play'][1] < quarters_frames[2][1]:
            row['quarter'] = '3'
        else:
            row['quarter'] = '4'
    return plays


def add_scoreboard(acta, plays):
    indx_posesionsA, indx_posesionsB, pointsA, pointsB = compute_points(acta)
    plays = plays.assign(pointsA='', pointsB='')
    for index, row in plays.iterrows():
        if index > 0:
            plays.loc[index-1,
                      'pointsA'] = int(compute_info_frame(row['play'][1], indx_posesionsA, pointsA))
            plays.loc[index-1,
                      'pointsB'] = int(compute_info_frame(row['play'][1], indx_posesionsB, pointsB))
    return plays


def add_partial(plays):
    plays = plays.assign(partial='')
    A = 0
    B = 0
    partial = 0
    for index, row in plays.iterrows():
        if index > 0 and row['pointsA'] != '' and row['pointsB'] != '':
            if row['pointsA'] != A:
                if partial <= 0:
                    partial = int(row['pointsA']) - A
                else:
                    partial = partial + int(row['pointsA']) - A
            if row['pointsB'] != B:
                if partial >= 0:
                    partial = -(row['pointsB'] - B)
                else:
                    partial = partial - (row['pointsB'] - B)
            A = row['pointsA']
            B = row['pointsB']
        row['partial'] = partial
    return plays


def add_team(plays):
    file = f'{ROOT_DIRECTORY}/player_position_{match}.txt'
    positions, dic_idx_frame = get_list_detections(file)
    positions = correct_positions(positions)
    dic_idx_frame = correct_diccionary(dic_idx_frame)
    for index, row in plays.iterrows():
        if row['play'][0] != None and row['team'] == '-':
            row['team'] = identify_team(positions, dic_idx_frame, row['play'], row['quarter'])

    return plays


def acta_get_plays(m, d):
    global match
    global ROOT_DIRECTORY
    match = m
    ROOT_DIRECTORY = d
    acta = pd.read_csv(f'{ROOT_DIRECTORY}/output_final.csv')
    plays = get_cuts(acta)
    plays = add_quarters(plays)
    plays = add_scoreboard(acta, plays)
    plays = add_players(acta, plays)
    plays = add_partial(plays)
    plays = add_team(plays)

    plays.to_csv(f'{ROOT_DIRECTORY}/plays_{match}.csv')
    return plays
