# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#  OlympIA pipeline for the video resume creation from the Acta file.
#  Designed and produced by the OlympIA team.
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from acta_preprocessing import acta_preprocessing
from acta_confidence_to_makes import acta_confidence_to_makes
from acta_sincronization import acta_sincronization
from acta_augmentation import acta_augmentation
from acta_get_plays import acta_get_plays
from acta_selection import acta_selection
from video_creation import video_creation

import os


def acta_olympia(match, directory):
    if not os.path.isfile(f'{directory}/plays_{match}.csv'):
        # Preprocessing of the acta file
        acta_preprocessing(match, directory)

        # Create the acta complete dataframe
        acta_confidence_to_makes(match, directory)
        acta_sincronization(match, directory)
        acta_augmentation(match, directory)

        # Do the subdividition in posesions
        acta_get_plays(match, directory)

    # Do the selection process with the user
    selected_dataframe = acta_selection(match, directory)
    selected_dataframe.to_csv(f'{directory}/selected_plays_{match}.csv', index=False, sep=';')

    # Create the video
    video_creation(selected_dataframe, f'{directory}/video_{match}.mkv')


acta_olympia(
    '1', "/Users/miquellopezescoriza/Documents/SISE QUATRIMESTRE/PROJECTE D'ENGINYERIA/ACTA_DIGITAL")
