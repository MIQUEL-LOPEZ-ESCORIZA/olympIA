# olympIA. An interactive basketball software for performace analysis. 

**Version 1.0.0**

OlympIA is publicly available software that uses a basketball game’s digital acta and video file to create an interactive session. This session allows users to search and select specific plays based on a wide range of features, such as the play outcome, three-point attempts, or missed shots.

---
# How to use it
To use this software one must obtain the video_file of the match and its digital game score sheet(which from now on we will call 'acta'). One must also obtain a file containing the player's positions at each frame of the video file. We have implemented a code using YOLO software to obtain this. However, we can't share it publicly due to privacy reasons. 

When having this files, save them into the same directory as the project under these names:

| File        | Name                         |
|-------------|------------------------------|
| Video file  | video_{match}.mkv            |
| Acta file   | ActaDigital_simplified_{match}.json |

When having these files saved, execute acta_olympia.py script. This is going to concatanate the execution of several scripts until an interactive session with the user appears. At this moment the user can select the plays from the match that he/she needs to visualize based on several criteria options. 

If you have different matches, save each match files with an integer subindex:{match}. 

---
# Implementation

This project is coded in Python and utilizes various libraries for data processing, visualization, and image handling.

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations and array handling.
- **OpenCV**: For image and video processing.
- **JSON**: For reading and writing JSON file formats.
- **CSV**: For handling CSV files.
- **Matplotlib**: For plotting and visualizations.
- **OS**: For interacting with the operating system.
- **Sys**: For system-specific parameters and functions.
- **AST**: For parsing and processing Python literals in strings.
- **PIL (Pillow)**: For image handling and processing.
- **Matplotlib Patches**: For adding shapes, such as rectangles and polygons, to plots.

Read the requirements.txt file to know the exact environment used when developing this project. 
 
---
# Code

Each script is thoroughly documented, providing clear instructions on how to use them. 









---
# Demos
Additionally, various files with examples showcase the capabilities of OlympIA.
