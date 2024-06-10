import pandas as pd
import numpy as np
import youtube_dl # client to download from many multimedia portals
import glob # directory operations
import os # interface to os-provided info on files
import shutil # interface to command line
from pydub import AudioSegment # only audio operations

def newest_mp3_filename():
    # lists all mp3s in local directory
    list_of_mp3s = glob.glob('./*.mp3')
    # returns mp3 with highest timestamp value
    return max(list_of_mp3s, key = os.path.getctime)

def get_video_time_in_ms(video_timestamp):
    vt_split = video_timestamp.split(":")
    if (len(vt_split) == 3): # if in HH:MM:SS format
        hours = int(vt_split[0]) * 60 * 60 * 1000
        minutes = int(vt_split[1]) * 60 * 1000
        seconds = int(vt_split[2]) * 1000
    else: # MM:SS format
        hours = 0
        minutes = int(vt_split[0]) * 60 * 1000
        seconds = int(vt_split[1]) * 1000
    # time point in miliseconds
    return hours + minutes + seconds

def get_trimmed(mp3_filename, initial, final = ""):
    if (not mp3_filename):
        # raise an error to immediately halt program execution
        raise Exception("No MP3 found in local directory.")
    # reads mp3 as a PyDub object
    sound = AudioSegment.from_mp3(mp3_filename)
    t0 = get_video_time_in_ms(initial)
    print("Beginning trimming process for file ", mp3_filename, ".\n")
    print("Starting from ", initial, "...")
    if (len(final) > 0):
        print("...up to ", final, ".\n")
        t1 = get_video_time_in_ms(final)
        return sound[t0:t1] # t0 up to t1
    return sound[t0:] # t0 up to the end


# downloads yt_url to the same directory from which the script runs
def download_audio(yt_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([yt_url])
        result = ydl.extract_info(yt_url, download=False)
        id = result['id']
    return id

def trim_audio(yt_url, initial, final, label, gender):
    id = download_audio(yt_url)
    
    initial_formatted = initial.replace(":", "")
    final_formatted = final.replace(":", "")
    
    
    filename = newest_mp3_filename()
    trimmed_file = get_trimmed(filename, initial, final)
    new_path = f"data_mp3/query/{id}_{label}_{gender}_{initial_formatted}_{final_formatted}.mp3"
    trimmed_file.export(new_path, format="mp3")
    # saves file with newer filename
    # shutil.copy(filename, new_path)
    print("Process concluded successfully. Saving trimmed file as ", new_path)
    os.remove(filename)
    return new_path


def find_duplicate_filenames(file_names):
    seen = {}
    duplicates = []

    for idx, file_name in enumerate(file_names):
        if file_name in seen:
            duplicates.append((seen[file_name], idx))
        else:
            seen[file_name] = idx

    return duplicates

if __name__ =='__main__':
    new_path = trim_audio(yt_url = 'https://www.youtube.com/watch?v=hbmf0bB38h0',initial = "00:15", final = "01:15", label = 'None', gender = 'None')
    print(new_path)

    # links = []
    # descriptions = []
    # filenames = []
    # notes = []
    # df = pd.read_csv('data_raw_query.csv')
    # for index, row in df.iterrows():
    #     link = row.iloc[1]
    #     start_time = row.iloc[2]
    #     end_time = row.iloc[3]
    #     description = row.iloc[4]
    #     label = row.iloc[5]
    #     gender = row.iloc[6]
    #     note  = row.iloc[7]
    #     print(link, start_time, end_time, description, label, gender, note)
    #     try:
    #         filename = trim_audio(link, start_time, end_time, label, gender).split('/')[-1]
    #         links.append(link)
    #         descriptions.append(description)
    #         filenames.append(filename)
    #         notes.append(note)
    #     except :
    #         continue

    # # Tạo DataFrame từ các danh sách
    # data = {'Link': links, 'Description': descriptions, 'Trimmed Filename': filenames, 'Note': notes}
    # df = pd.DataFrame(data)
    # # Ghi DataFrame vào file CSV
    # df.to_csv('datasets_mp3_query.csv', index=False, encoding='utf-8') 


    # filenames = os.listdir('data_mp3') # 128
    # filenames = []
    # df = pd.read_csv('datasets_mp3.csv')
    # for index, row in df.iterrows():
    #     filename = row.iloc[2]
    #     filenames.append(filename)
    #     # if filename not in filenames:
    #         # print(filename)
    # duplicates = find_duplicate_filenames(filenames)
    # print(duplicates) # [(36, 37), (61, 62)]
