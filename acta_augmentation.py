# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This OlympIA script is the flagship product, designed to extract key metrics 
# from a .txt file containing the positions of all players detected by YOLO software at each frame. 
# Using these metrics—such as centroid velocity and player positions—it enriches the match report 
# with detailed contextual data. For instance, it identifies fouls by detecting sudden drops in total 
# velocity, among other indicators.
# Designed and produced by the OlympIA team.
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pandas as pd
import numpy as np
import os
import cv2
import json
import csv
from pandas import json_normalize
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


''' This function converts the detection file of positions into a vector of positions and a dictionary to its indexes.
    positions = [frame, [positions]] and a dictionary = {frame:index in positions} '''


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


''' These two functions are complementary to get_list_detections. '''


def get_detections(positions, dic_idx_frame, frame):
    return positions[dic_idx_frame[frame]][-1]


def number_of_detections(positions, dic_idx_frame, frame):
    if frame in dic_idx_frame:
        return len(get_detections(positions, dic_idx_frame, frame))
    else:
        return 0


''' This function deletes all the useless events in the acta. The end and start of period
    have already been detected with more precision so there is no need to use an approximation. '''


def acta_delete_non_important_events(acta, start_frame, end_frame):
    df = acta.copy()
    df = df[df['timestamp'] >= start_frame]
    df = df[df['description'] != 'inicio de partido']
    df = df[df['description'] != 'fin de partido']
    df = df[df['description'] != 'inicio periodo']
    df = df[df['description'] != 'fin de periodo']
    df = df[df['description'] != 'recuperación inicial']
    df = df[df['description'] != 'pérdida']
    return df


''' This function corrects the positions of the players, for some frames there
    exist no positions so it creates a null vector for these. '''


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


''' This function corrects the diccionary of the frames, for some frames there
    exist no positions so we must add them to the dictionary. '''


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


''' This function returns a vector with the absolute mean of the velocity of the centroid of the
    players for each frame. As we don't know which position of each player corresponds to the position
    of the same player in the next frame, we calculate the one with minimum distance.'''


def calculate_absolute_mean_velocity(positions, dic_idx_frame):

    total_velocitats = []
    frames = []

    for indx in range(len(positions)-1):
        frames.append(positions[indx][0])
        positions_frame_1 = positions[indx][1]
        positions_frame_2 = positions[indx+1][1]
        velocitats = []
        for positions_1 in positions_frame_1:
            min_dist = np.inf
            for positions_2 in positions_frame_2:

                distx = positions_1[1] - positions_2[1]
                disty = positions_1[2] - positions_2[2]

                dist = np.sqrt((distx*15/28)**2 + (disty)**2)

                if dist < min_dist:
                    min_dist = dist
            if min_dist < 0.007:
                velocitats.append(min_dist)

        if (len(velocitats) > 5):
            total_velocitats.append(sum(velocitats)/len(velocitats))
        else:
            total_velocitats.append(0)

    window_size = 250  # the size of the moving average window
    window = np.ones(window_size) / float(window_size)
    total_velocitats = np.convolve(total_velocitats, window, 'same')

    return total_velocitats


''' This functions adds to the acta a row when there is a change of the attacking team.
    This is done by calculating when the centroid of the players crosses the middle line of
    the court. '''


def acta_add_change_of_defenders(acta, positions):
    df = acta.copy()
    mean_values = []
    frame_values = []

    # loop over the range of file numbers
    for element in positions:
        frame = element[0]
        values = [a[1] for a in element[1]]
        if len(values) != 0:
            mean = sum(values) / len(values)
        else:
            mean = None

        mean_values.append(mean)
        frame_values.append(frame)

    crossings = []
    for i in range(1, len(mean_values)):
        if mean_values[i] != None and mean_values[i-1] != None:
            if ((mean_values[i-1] - 0.5) * (mean_values[i] - 0.5)) <= 0:
                crossings.append(frame_values[i])

    # Filtrate crossings: We only want those where the change of posesion lasts more than a second
    filtrated_crossings = []
    for i in range(0, len(crossings)):
        if i < len(crossings) - 1:
            if crossings[i+1] - crossings[i] > 100:
                filtrated_crossings.append(crossings[i])
        else:
            if crossings[i] - filtrated_crossings[-1] > 100:
                filtrated_crossings.append(crossings[i])

    for i in range(len(filtrated_crossings)):
        timestamp = filtrated_crossings[i]
        idx = (df['timestamp'] >= timestamp).idxmax()
        df = pd.concat([df.iloc[:idx], pd.DataFrame({'timestamp': [timestamp], 'description': [
            "cambio atacante"]}), df.iloc[idx:]]).reset_index(drop=True)
    return df


''' This function returns a vector containing the timestamps where there is a foul. '''


def calculate_frames_of_fouls(df, end_frame):
    # Select the rows where the description is 'personal'
    personal_df = df[(df['description'] == 'personal') | (df['description'] == 'personal 2tl') | (
        df['description'] == 'personal 3tl') | (df['description'] == 'personal 1tl')]

    # Extract the timestamp column from the personal_df DataFrame
    personal_timestamps = personal_df['timestamp'].values
    filtered_timestamps = [t for t in personal_timestamps if t < end_frame]

    return filtered_timestamps


''' This function returns a vector of timestamps that indicate when the game has
    been restarted after a foul. This is done by searching when the absolute mean
    velocity of the centroid crosses a boundary value. '''


def calculate_frames_of_game_restart_after_fouls(start_frame, total_velocitats, timestamps):
    indexes = []
    for timestamp in timestamps:
        index = timestamp - start_frame + 250
        while total_velocitats[index] < 0.00122:
            index = index + 1
        indexes.append(index + start_frame)
    return indexes


''' This function returns a vector containing the timestamps where there is a timeout. '''


def calculate_frames_of_timeouts(df, end_frame):
    # Select the rows where the description is 'personal'
    personal_df = df[(
        df['description'] == 'tiempo muerto')]

    # Extract the timestamp column from the personal_df DataFrame
    personal_timestamps = personal_df['timestamp'].values
    filtered_timestamps = [t for t in personal_timestamps if t < end_frame]

    return filtered_timestamps


''' This function returns a vector of timestamps that indicate when the game has
    been restarted after a timeout. This is done by searching when the absolute mean
    velocity of the centroid crosses a boundary value. '''


def calculate_frames_of_game_restart_after_timeouts(start_frame, total_velocitats, timestamps):
    indexes = []
    for timestamp in timestamps:
        index = timestamp - start_frame + 500
        while total_velocitats[index] < 0.00122:
            index = index + 1
        indexes.append(index + start_frame)
    return indexes


''' This function adds a row to acta which timestamp indicates the game restart after the foul. '''


def acta_add_game_restart_after_fouls(indexes, acta):
    df = acta.copy()
    for timestamp in indexes:
        idx = (df['timestamp'] >= timestamp).idxmax()
        df = pd.concat([df.iloc[:idx], pd.DataFrame({'timestamp': [timestamp], 'description': [
                       "reinicio despues de falta"]}), df.iloc[idx:]]).reset_index(drop=True)
    return df


''' This function adds a row to acta which timestamp indicates the game restart after the timeout. '''


def acta_add_game_restart_after_timeouts(indexes, acta):
    df = acta.copy()
    for timestamp in indexes:
        idx = (df['timestamp'] >= timestamp).idxmax()
        df = pd.concat([df.iloc[:idx], pd.DataFrame({'timestamp': [timestamp], 'description': [
                       "reinicio despues de timeout"]}), df.iloc[idx:]]).reset_index(drop=True)
    return df


''' This function returns a vector of intervals. Each of these intervals indicate
    when the game has been stopped. This is done by calculating when the absolute mean
    velocity of the centroid is kept under a certian boundary velocity for a certain boundary time. '''


def calculate_frames_of_slow_game_pace(start_frame, total_velocitats):
    slow_pace_timestamps = []

    index = 0
    count = 0

    interval_now = []
    while index < len(total_velocitats):
        velocity = total_velocitats[index]
        if velocity < 0.00117:
            count += 1
            if count == 1:
                interval_now.append(index + start_frame)
        else:
            if count >= 450:
                interval_now.append(index + start_frame)
                slow_pace_timestamps.append(interval_now)
            count = 0
            interval_now = []
        index += 1
    return slow_pace_timestamps


''' This function deletes the game stopping intervals which are due to fouls or timeouts. '''


def acta_set_right_clocks_on_and_off(clock_on_off_vector, frames):
    for frame in frames:
        min_diff = float('inf')
        nearest_vector = None
        nearest_index = None
        for i, vector in enumerate(clock_on_off_vector):
            diff = abs(frame - vector[0])
            if diff < min_diff:
                min_diff = diff
                nearest_vector = vector
                nearest_index = i
            else:
                break  # Since the vectors are sorted, we can stop searching if the difference starts increasing

        if nearest_vector is not None:
            del clock_on_off_vector[nearest_index]
    return clock_on_off_vector


''' This function adds to acta the intervals when the game has been stopped. '''


def acta_add_clock_on_off(clock_on_off_vector, acta):
    df = acta.copy()
    for timestamps in clock_on_off_vector:
        timestamp_1 = timestamps[0]
        timestamp_2 = timestamps[1]
        idx = (df['timestamp'] >= timestamp_1).idxmax()
        df = pd.concat([df.iloc[:idx], pd.DataFrame({'timestamp': [timestamp_1], 'description': [
                       "clock_off"]}), df.iloc[idx:]]).reset_index(drop=True)
        idx = (df['timestamp'] >= timestamp_2).idxmax()
        df = pd.concat([df.iloc[:idx], pd.DataFrame({'timestamp': [timestamp_2], 'description': [
                       "clock_on"]}), df.iloc[idx:]]).reset_index(drop=True)
    return df


''' This function acts as a repair function. If there is any important event inside an stopping interval we separate
    it in two cases:

        1 - The important event is very close to the clock-off timestamp
          - For this case we move the important event outside the stopping interval
            as it is due to a mistake of sincronization.

        2 - The important event is not close to the clock-off timestamp
          - There has been a mistake on this stopping interval. The game pace is slow
            but is not a clock_off situation so we delete this stopping interval.
'''


def acta_repair_mistakes(acta):
    df = acta.copy()
    important_events = ['intento fallado de 2',
                        'intento fallado de 3', 'canasta de 2', 'canasta de 3']
    new_rows = []

    for index, row in df.iterrows():
        if row['description'] == 'clock_off':
            timestamp_clock_off = row['timestamp']
            next_row_index = index + 1
            delete_clocks = False

            while df.loc[next_row_index]['description'] != 'clock_on':
                next_row = df.loc[next_row_index]
                if next_row['description'] in important_events:
                    if next_row['timestamp'] - timestamp_clock_off < 200:
                        new_row = next_row.copy()
                        new_row['timestamp'] = timestamp_clock_off - 50
                        new_rows.append(new_row)
                        df.drop(next_row_index, inplace=True)

                    else:
                        delete_clocks = True
                next_row_index += 1

            if delete_clocks:
                df.drop([index, next_row_index], inplace=True)

    for new_row in new_rows:
        idx = (df['timestamp'] >= new_row['timestamp']).idxmax()
        df = pd.concat([df.iloc[:idx], pd.DataFrame(new_row).T,
                        df.iloc[idx:]]).reset_index(drop=True)
    return df


''' All of the following functions act with the same aim. Their objective is to prepare an
    optimal dataframe to be capable to separate the game in plays without errors. '''

''' This function deletes the change of attacking team right after an event in events(after timeout restart,
    after foul restart, after quarter restart). This is done because the change of attacking team does not represent the end of any play so it
    is deleted. '''


def acta_delete_changes(acta, events):
    df = acta.copy()
    i = 0
    while i < len(df)-1:
        if df['description'][i] in events:
            if df['description'][i+1] == 'cambio atacante':
                df = df.drop(i+1).reset_index(drop=True)
        i += 1
    return df


''' This function returns a vector with the start and end frames of each period
    from the file cuts made in the detections.'''


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


''' This function adds the start-end quarters timestamps detected to acta. It also deletes the
    change of attacking team right after the start of quarter as it does not indicate a play has ended. '''


def acta_add_quarters_and_delete_changes(acta):
    df = acta.copy()
    quarters = cuts_quarters()
    idx = (df['timestamp'] >= quarters[3][1]).idxmax()

    df = pd.concat([df.iloc[:idx], pd.DataFrame({'timestamp': quarters[3][1], 'description': [
        "fin partido"]}), df.iloc[idx:]]).reset_index(drop=True)
    for quarter in quarters:
        idx = (df['timestamp'] >= quarter[0]).idxmax()
        df = pd.concat([df.iloc[:idx], pd.DataFrame({'timestamp': quarter[0], 'description': [
            "inicio periodo"]}), df.iloc[idx:]]).reset_index(drop=True)
        if df['description'][idx+1] == 'clock_on':
            df = df.drop(idx+1).reset_index(drop=True)
        idx2 = (df['timestamp'] >= quarter[1]).idxmax()
        df = pd.concat([df.iloc[:idx2], pd.DataFrame({'timestamp': quarter[1], 'description': [
            "fin periodo"]}), df.iloc[idx2:]]).reset_index(drop=True)
        if df['description'][idx2-1] == 'clock_off':
            df = df.drop(idx2-1).reset_index(drop=True)

    df = pd.concat([df.iloc[:0], pd.DataFrame({'timestamp': quarters[0][0], 'description': [
        "inicio partido"]}), df.iloc[0:]]).reset_index(drop=True)
    return df


''' This function deletes the change of attacking team between a the time a foul/timeout/end of period and the time
    the game restarts. This is done because the change of attacking team does not represent the end of any play so it
    is deleted. '''


def acta_delete_changes_between_timeouts_faults_quarters(acta):
    df = acta.copy()
    i = 0
    while i < len(df):
        if df['description'][i] == 'personal' or df['description'][i] == 'personal 2tl' or df['description'][i] == 'personal 3tl' or df['description'][i] == 'personal 1tl' or df['description'][i] == 'tiempo muerto' or df['description'][i] == 'fin periodo':
            while i < len(df) and (df['description'][i] != 'reinicio despues de falta' and df['description'][i] != 'reinicio despues de timeout' and df['description'][i] != 'inicio periodo'):
                if df['description'][i] == 'cambio atacante':
                    df = df.drop(i).reset_index(drop=True)
                else:
                    i += 1
        i += 1
    return df


def acta_augmentation(m, d):
    global match
    global ROOT_DIRECTORY
    match = m
    ROOT_DIRECTORY = d

    acta = pd.read_csv(f'{ROOT_DIRECTORY}/offset_acta_{match}vdef.csv')

    positions, dic_idx_frame = get_list_detections(
        f'{ROOT_DIRECTORY}/player_position_{match}.txt')

    start_frame = positions[0][0]
    end_frame = positions[len(positions)-1][0]

    acta = acta_delete_non_important_events(acta, start_frame, end_frame)

    positions = correct_positions(positions)
    dic_idx_frame = correct_diccionary(dic_idx_frame)

    velocities = calculate_absolute_mean_velocity(positions, dic_idx_frame)

    acta = acta_add_change_of_defenders(acta, positions)

    frames_of_fouls = calculate_frames_of_fouls(acta, end_frame)
    frames_of_game_restart = calculate_frames_of_game_restart_after_fouls(
        start_frame, velocities, frames_of_fouls)
    acta = acta_add_game_restart_after_fouls(frames_of_game_restart, acta)

    frames_of_timeouts = calculate_frames_of_timeouts(acta, end_frame)
    frames_of_game_restart = calculate_frames_of_game_restart_after_timeouts(
        start_frame, velocities, frames_of_timeouts)
    acta = acta_add_game_restart_after_timeouts(frames_of_game_restart, acta)

    clock_on_off_vector = calculate_frames_of_slow_game_pace(start_frame, velocities)
    clock_on_off_vector = acta_set_right_clocks_on_and_off(
        clock_on_off_vector, frames_of_fouls + frames_of_timeouts)
    acta = acta_add_clock_on_off(clock_on_off_vector, acta)

    acta = acta_repair_mistakes(acta)
    acta = acta_delete_changes(acta, ['clock_on', 'reinicio despues de falta',
                                      'reinicio despues de timeout'])
    acta = acta_delete_changes_between_timeouts_faults_quarters(acta)

    acta = acta_add_quarters_and_delete_changes(acta)

    acta.to_csv(f'{ROOT_DIRECTORY}/output_final.csv', index=False)

