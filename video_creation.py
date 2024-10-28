# Import the necessary libraries
import pandas as pd
import numpy as np
import os
import subprocess
import cv2


def extract_frames(video_path, timestamps, output_path):
    input_files = []
    concat_file = 'concat.txt'

    # Extract individual clips and create a list of input files
    for i, (start, end) in enumerate(timestamps):
        clip_output_path = f'clip{i}.mkv'
        cmd = ['ffmpeg', '-i', video_path, '-ss',
               str(start), '-to', str(end), '-c', 'copy', clip_output_path]
        subprocess.run(cmd)
        input_files.append(clip_output_path)

    # Generate a concatenation file with the list of input files
    with open(concat_file, 'w') as f:
        for file in input_files:
            f.write(f"file '{file}'\n")

    # Concatenate the clips using ffmpeg
    cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file, '-c', 'copy', output_path]
    subprocess.run(cmd)

    # Remove temporary files
    # Remove temporary files
    for file in input_files:
        os.remove(file)
    os.remove(concat_file)


def video_creation(selected_df, video_file):
    timestamps = []

    for index, row in selected_df.iterrows():
        event = row['final_event']
        plays_names_violation = ['violation', 'shot_miss_2 + violation', 'shot_miss_3 + violation']
        plays_names_shots = ['shot_made_2', 'shot_made_3']
        start, end = row['play']

        if event in plays_names_violation:
            timestamps.append([(int(start) - 100)/50, (int(end) + 500)/50])
        elif event in plays_names_shots:
            timestamps.append([(int(start) - 100)/50, (int(end) + 100)/50])
        else:
            timestamps.append([(int(start) - 100)/50, (int(end) + 300)/50])

    print("Producing video...")

    extract_frames(video_file, timestamps, 'extracted_video.mkv')
