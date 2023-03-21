"""
    Run this script to shuffle names of files in folder 
"""

from dataset import YoloData
import datetime, random, os 

def shuffle_file_names(folder_name : str) -> None:
    temporary_filename = os.path.join(folder_name, datetime.datetime.now().strftime("temporary_filename_%y%m%d_%H%M%S"))
    filenames = YoloData.find_files_in_folder(folder_name)
    shuffled_filenames = [ x for x in filenames ]
    random.shuffle(shuffled_filenames)
    for idx, filename in enumerate(filenames):
        os.rename(filename, f"{temporary_filename}_{idx}")
    for idx, filename in enumerate(filenames):
        os.rename(f"{temporary_filename}_{idx}", shuffled_filenames[idx])

if (__name__ == "__main__"):

    target_folder = os.path.join(os.path.dirname(__file__), "../wizard101 screenshots/wizard")

    shuffle_file_names(target_folder)