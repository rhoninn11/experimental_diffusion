
import os
import shutil
from tqdm import tqdm

def is_png(file: str):
    if file.endswith(".png"):
        return True
    
    return False

def decimate_results():
    path = "result"
    out_path = f"{path}/decimated"
    files = os.listdir(path)
    png_files = list(filter(is_png, files))
    png_files = png_files[::4] #decimate

    for i, file in tqdm(enumerate(png_files)):
        new_file = f"train_{i:05d}.png"
        file_from = f"{path}/{file}"
        file_to = f"{out_path}/{new_file}"
        shutil.copy(file_from, file_to)
        # print(new_file)
        
decimate_results()
