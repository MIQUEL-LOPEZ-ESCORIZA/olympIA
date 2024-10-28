# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#  OlympIA script that enables the user to select the possessions
#  he wants from the dataframe created with the Acta pipeline.
#  Designed and produced by the OlympIA team.
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from image_court import generate_image
import pandas as pd
import numpy as np
import sys
import ast


def filtrate_team(df, team):
    return df[df['team'] == team]


def filtrate_quarter(df, quarters):

    for num in quarters:
        if num not in ['1', '2', '3', '4']:
            print("The quarters must be 1, 2, 3 and/or 4!")
            sys.exit(1)
    quarters = [int(a) for a in quarters]

    return df[(df['quarter'].isin(quarters))]


def filtrate_plays(df):
    print("These are the types of play available:\n")
    print('1- Steals, 2- Violations, 3- Missed shots of 2, 4- Missed shots of 3, 5- Two-Pointers, 6- Three-Pointers, 7- Fouls in play, ' +
          '8- Missed shot of 2 + violation, 9- Missed shot of 3 + violation, 10- Missed shot of 2 + foul, 11- Missed shot of 3 + foul,  11- End of quarter \n')
    plays = input("Enter type of plays wanted separated by commas: \n").split(",")
    plays = [int(a) for a in plays]

    plays_names = ['steal', 'violation', 'shot_miss_2', 'shot_miss_3',
                   'shot_made_2', 'shot_made_3', 'foul', 'shot_miss_2 + violation',
                   'shot_miss_3 + violation', 'shot_miss_2 + foul', 'shot_miss_3 + foul', 'fin de periodo']
    plays_names_selected = []
    for i in range(len(plays_names)):
        if i+1 in plays:
            plays_names_selected.append(plays_names[i])

    return df[df['final_event'].isin(plays_names_selected)]


def filtrate_plays_before(df_initial, team):
    # Get the input
    print("These are the types of play available:\n")
    print('1- Steals, 2- Violations, 3- Missed shots of 2, 4- Missed shots of 3, 5- Two-Pointers, 6- Three-Pointers, 7- Fouls in play, ' +
          '8- Missed shot of 2 + violation, 9- Missed shot of 3 + violation, 10- Missed shot of 2 + foul, 11- Missed shot of 3 + foul,  11- End of quarter \n')
    plays_before = input(
        "Enter the ending event of the play before the wanted ones separated by commas:").split(",")
    plays_before = [int(a) for a in plays_before]

    plays_names = ['steal', 'violation', 'shot_miss_2', 'shot_miss_3',
                   'shot_made_2', 'shot_made_3', 'foul', 'shot_miss_2 + violation',
                   'shot_miss_3 + violation', 'shot_miss_2 + foul', 'shot_miss_3 + foul', 'fin de periodo']
    plays_names_selected = []
    for i in range(len(plays_names)):
        if i+1 in plays_before:
            plays_names_selected.append(plays_names[i])

    if team != 'both':
        return df_initial[(df_initial['final_event'].shift(1).isin(plays_names_selected)) & (df_initial['team'] == team)]
    else:
        return df_initial[df_initial['final_event'].shift(1).isin(plays_names_selected)]


def filtrate_after_timeout(df):
    return df[df['after_timeout']]


def filtrate_players(df, team):
    if team == 'both':
        print("Error: you must choose an attacking team in order to filter by players on the court!")
        return df
    else:
        # Get the input
        players = input("Enter the players' numbers wanted separated by commas:").split(",")
        players = [int(a) for a in players]

        # Create a set of the players to be searched for
        players_set = set(players)

        # Create a boolean mask to select the rows that match the criteria
        if team == 'A':
            mask = df['playersA'].apply(lambda x: x is not None and players_set.issubset(set(x)))
        elif team == 'B':
            mask = df['playersB'].apply(lambda x: x is not None and players_set.issubset(set(x)))
        else:
            raise ValueError("Invalid team value. It should be 'A' or 'B'.")

        return df[mask]


def filtrate_parcial(df, team):
    input_str = input("Enter if you want the partial to be positive (p) or negative (n) for the atacking team being studied and then " +
                      "the limit of the partial: ")
    negative, partial = input_str.split()
    if negative == 'n':
        negative = True
    else:
        negative = False
    partial = int(partial)

    if (negative and team == 'A') or (not negative and team == 'B'):
        return(df[df['partial'] <= -partial])
    elif (negative and team == 'B') or (not negative and team == 'A'):
        return(df[df['partial'] >= partial])


def filtrate_actors(df):
    actors = input(
        "Enter actors' ids as they appear in the game_legend file wanted separated by commas:").split(",")
    return df[(df['actor'].isin(actors))]


def df_types(df):
    df['partial'] = df['partial'].astype('int')
    df['team'] = df['team'].astype('str')
    df['final_event'] = df['final_event'].astype('str')
    df['after_timeout'] = df['after_timeout'].astype('boolean')
    return df


def selection_process(complete_dataframe):
    df = df_types(complete_dataframe)
    initial_df = df.copy()
    resulting_df = df.copy()

    # Filter by team
    team = input("Which team's atacking plays do you want to watch? ('A', 'B' or 'both')")
    if team in ['A', 'B']:
        df = filtrate_team(df, team)
    elif team != 'both':
        print("The team must be 'A', 'B' or 'both'!")
        sys.exit(1)

    # Filter by quarter
    quarters = input(
        "Do you want to watch all plays (write 'all') or filter by quarters (write the wanted quarters separated by commas)?").split(",")
    if len(quarters) >= 1 and quarters[0] != 'all':
        df = filtrate_quarter(df, quarters)

    # User selection process
    num = '0'
    initial = True
    while(num != 'done'):
        print("\nWrite the number of the feature you want to filter by, followed by an 'r' if you want to do it in a Restrictive Mode.")
        print("(Restrictive Mode: all plays MUST satisfy your filtering condition.)\n")

        print("1- Type(s) of play")
        print("2- Ending type(s) of the previous play")
        print("3- Actor(s)")
        print("4- Player(s) on the court")
        print("5- Scoring partial")
        print("6- Plays after timeout")

        print("Write 'done' if you want to finish the play selection process.\n")

        if initial:
            print(str(len(df)) + ' plays available\n')
        else:
            print(str(len(resulting_df)) + ' plays selected     -     ' +
                  str(len(df)) + ' plays available\n')

        selection = input("Write here:").split()
        num = selection[0]
        if len(selection) > 1:
            mode = 'R'
        else:
            mode = 'N'

        if num != 'done':
            if initial:
                if mode == 'R':
                    resulting_df = df.copy()
                else:
                    resulting_df = pd.DataFrame(columns=initial_df.columns)
                initial = False
            if num == '1':
                if mode == 'R':
                    resulting_df = filtrate_plays(resulting_df)
                else:
                    resulting_df = pd.concat(
                        [resulting_df, filtrate_plays(df)]).reset_index(drop=True)
            elif num == '2':
                if mode == 'R':
                    aux_df = filtrate_plays_before(initial_df, team)
                    resulting_df = pd.merge(resulting_df, aux_df, on='play', how='inner', suffixes=(
                        '', '_other')).reset_index(drop=True)
                    resulting_df = resulting_df.loc[:, aux_df.columns]
                else:
                    resulting_df = pd.concat([resulting_df, filtrate_plays_before(
                        initial_df, team)]).reset_index(drop=True)
            elif num == '3':
                if mode == 'R':
                    resulting_df = filtrate_actors(resulting_df)
                else:
                    resulting_df = pd.concat(
                        [resulting_df, filtrate_actors(df)]).reset_index(drop=True)
            elif num == '4':
                if mode == 'R':
                    resulting_df = filtrate_players(resulting_df, team)
                else:
                    resulting_df = pd.concat(
                        [resulting_df, filtrate_players(df, team)]).reset_index(drop=True)
            elif num == '5':
                if mode == 'R':
                    resulting_df = filtrate_parcial(resulting_df, team)
                else:
                    resulting_df = pd.concat(
                        [resulting_df, filtrate_parcial(df, team)]).reset_index(drop=True)
            elif num == '6':
                if mode == 'R':
                    resulting_df = filtrate_after_timeout(resulting_df)
                else:
                    resulting_df = pd.concat(
                        [resulting_df, filtrate_after_timeout(df)]).reset_index(drop=True)

            resulting_df = resulting_df.drop_duplicates()

    resulting_df.loc[:, 'play'] = resulting_df['play'].apply(ast.literal_eval)
    df_sorted = resulting_df.sort_values('play')

    image_answer = input(
        'Do you want to create an image with the shots made during the match?(y/n)')
    if image_answer == 'y':
        generate_image(match, ROOT_DIRECTORY)

    return df_sorted


def acta_selection(m, d):
    global match
    global ROOT_DIRECTORY
    match = m
    ROOT_DIRECTORY = d

    plays = pd.read_csv(f'{ROOT_DIRECTORY}/plays_{match}.csv')

    df_selected_plays = selection_process(plays)

    df_selected_plays.to_csv(f'{ROOT_DIRECTORY}/plays_{match}_total_prova_selected.csv')

    return df_selected_plays
