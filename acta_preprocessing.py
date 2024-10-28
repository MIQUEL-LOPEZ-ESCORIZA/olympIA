# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#  OlympIA script that This script processes sports event data by reading
#  associated video and JSON metadata (digital acta), enriching it with
#  additional details, and then exporting the data into a separate CSV file.
#  Designed and produced by the OlympIA team.
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++








# Import the necessary libraries
import pandas as pd
import numpy as np
import cv2
import json
import csv
from pandas import json_normalize


# Given a match corresponding to a video in our database, returns a json containing all in 
# the information in the digital acta and the value corresponding to the frames_per_second
def read_files(match):
    # Reading of the video to compute the frames_per_second value
    file = f'{ROOT_DIRECTORY}/ActaDigital_simplified_{match}.json'
    video = cv2.VideoCapture(f'{ROOT_DIRECTORY}/Video_{match}.mkv')
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Reading the json file (the digital acta)
    with open(file, 'r') as f:
        json_acta = json.load(f)
    return json_acta, fps


# Given a timestamp (yyymmddhhmmss) and the frames_per_second values, returns the associated integer number of frames
def transform_to_frames(timestamp, fps):
    hour = int(timestamp[-6:-4])
    minute = int(timestamp[-4:-2])
    second = int(timestamp[-2:])
    total = int((hour*3600+minute*60+second)*fps)
    return total


def change_event_info(df_events):
    # Read the event types document as a data frame
    event_type = pd.read_excel(f'{ROOT_DIRECTORY}/event_types_acta_digital.xlsx')
    event_type = event_type.loc[:, ['uuid', 'description']]
    event_type = event_type.rename(columns={'uuid': 'eventTypeUuid'})

    # Convert all strings values to lowercase
    event_type['description'] = event_type['description'].str.lower()

    # Join the types and the events data frames
    df_events = pd.merge(df_events, event_type, on='eventTypeUuid', how='inner')
    df_events = df_events.drop(['eventTypeUuid'], axis=1)
    df_events = df_events.sort_values(by=['timestamp'])
    return df_events


def change_team_info(df_events, dic_game, df_game):
    id_teamA = df_game['homeTeamUuid'][0]
    id_teamB = df_game['awayTeamUuid'][0]
    teamA = df_game['homeTeamName'][0]
    teamB = df_game['awayTeamName'][0]

    # Add team's information to the game legend
    dic_game['A'] = teamA.lower()
    dic_game['B'] = teamB.lower()

    # Replace identifiers in the list of events
    df_events = df_events.replace({id_teamA: 'A', id_teamB: 'B'})
    df_events = df_events.rename(columns={'teamUuid': 'team'})
    return df_events, dic_game, df_game


def change_player_info(df_events, dic_game, df_actors):
    df_actors = df_actors.loc[:, ['actorUuid', 'actorName']]

    # Consider the full list of the players and add an id 
    for player in range(len(df_actors)):
        dic_game[player] = df_actors['actorName'][player]
        df_actors['actorName'][player] = player

    # Join the players and the events data frames
    df_events = pd.merge(df_events, df_actors, on='actorUuid', how='left')
    df_events = df_events.drop(['actorUuid'], axis=1)
    df_events = df_events.rename(columns={'actorName': 'actor'})
    return df_events, dic_game, df_actors


def add_scoreboard_info(df_events):
    df_events.insert(9, 'A', np.zeros(len(df_events)), True)
    df_events.insert(10, 'B', np.zeros(len(df_events)), True)
    df_events['A'] = df_events['A'].astype('int32')
    df_events['B'] = df_events['B'].astype('int32')

    # Scan the full list of events and add the score to each team
    df_events = df_events.replace({'canasta de tiro libre': 'canasta de 1'})
    pointsA = 0
    pointsB = 0
    for event in range(len(df_events)):
        if df_events['description'][event][0:10] == 'canasta de'  or df_events['description'][event]=='mate':
            # Identify 
            points = df_events['description'][event][-1]
            if points == 'e': points = 2
            else: points = int(points)

            if df_events['team'][event] == 'A':
                df_events.iloc[event,9] = pointsA + points
                df_events.iloc[event,10] = pointsB
                pointsA += points
            elif df_events['team'][event] == 'B':
                df_events.iloc[event,9] = pointsA
                df_events.iloc[event,10] = pointsB + points
                pointsB += points
        else:
            df_events.iloc[event,9] = pointsA
            df_events.iloc[event,10] = pointsB
    return df_events


def change_timestamp_info(df_events, fps):
    first_event = transform_to_frames(df_events['timestamp'][0], fps)

    # Compute the difference for each event on the list
    for i in range (len(df_events)):
        df_events.loc[i, 'timestamp'] = transform_to_frames(str(df_events.loc[i, 'timestamp']), fps) - first_event

    return df_events


def acta_preprocessing(m, d):
    global match
    global ROOT_DIRECTORY
    match = m
    ROOT_DIRECTORY = d
    # Read the acta and the video associated to a match
    json_acta, fps = read_files(match)

    # Divide the information kept in the json into 3 different data frames
    df_game = json_normalize(json_acta['gameInfo'])     
    df_actors = json_normalize(json_acta['actorsInfo'])
    df_events = json_normalize(json_acta['eventsInfo'])

    # Creation of a dictionary to save the general information of the match
    dic_game = {}
    dic_team_a = {}
    dic_team_b = {}
    dic_game['category'] = df_game['categoryName'][0].lower()
    dic_game['competition'] = df_game['competitionName'][0].lower()
    dic_game['day'] = f'{df_events["timestamp"][0][6:8]}-{df_events["timestamp"][0][4:6]}-{df_events["timestamp"][0][0:4]}'

    # Convert all strings values to lowercase
    df_actors['actorName'] = df_actors['actorName'].str.lower()

    # Do the modifications in the events df and generate the gamelegend
    df_events = change_event_info(df_events)
    df_events, dic_game, df_game = change_team_info(df_events, dic_game, df_game)
    df_events, dic_game, df_actors = change_player_info(df_events, dic_game, df_actors)
    df_events = add_scoreboard_info(df_events)
    df_events = change_timestamp_info(df_events, fps)

    # Fill the blanck spaces with None values
    df_events = df_events.fillna(np.nan).replace([np.nan], [None])

    # Export new acta (list of events and game legend)
    df_events.to_csv(f'{ROOT_DIRECTORY}/events_{match}.csv', index=False)
    with open(f'{ROOT_DIRECTORY}/game_legend_{match}.csv', 'w') as f: 
        w = csv.DictWriter(f, dic_game.keys())
        w.writeheader()
        w.writerow(dic_game)


#acta_preprocessing()

