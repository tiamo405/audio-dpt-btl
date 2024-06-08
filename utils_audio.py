import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
from scipy.signal import resample
import os

def read_mp3_by_AudioSegment(file_path):
    audio = AudioSegment.from_mp3(file_path)
    # Chuyển đổi âm thanh thành mảng numpy
    audio_array = np.array(audio.get_array_of_samples())
    # Lấy tần số lấy mẫu
    sample_rate = audio.frame_rate
    return audio_array, sample_rate


def read_audio_by_wavfile(file_path, sr):
    sample_rate, audio_data = wavfile.read(file_path)
    if sample_rate != sr:
        audio_data = resample(audio_data[:, 0], int(len(audio_data) * sr / sample_rate))
    if audio_data.ndim == 2:
        audio_data = audio_data[:, 0]
    return audio_data, sr 

def read_audio_by_librosa(file_path, sr):
    audio_data, sr = librosa.load(file_path, sr=sr)

    return audio_data, sr


def convert_mp3_to_wav(file_path, output):
    audio = AudioSegment.from_mp3(file_path)

    # Lưu file âm thanh dưới dạng wav
    audio.export(output, format="wav")




if __name__ == '__main__':

    # Ngưỡng là 5% của giá trị biên độ lớn nhất
    # threshold_percent = 0.05
    # sample_rate, denoised_audio = remove_noise(file_path= 'wavfiles/Acoustic_guitar/0eeaebcb.wav', threshold_percent= threshold_percent)


    # # Lưu lại file âm thanh đã lọc nhiễu
    # filtered_file_path = 'denoised_audio_file.wav'
    # wavfile.write(filtered_file_path, sample_rate, denoised_audio.astype(np.int16))

    # audio_array, sample_rate = read_mp3('data/-4SrLmSxCLg_speech_female_0000_0130.mp3')
    # y, sr = librosa.load('data/-4SrLmSxCLg_speech_female_0000_0130.mp3', sr= None, dtype = None)
    
    filenames = os.listdir('data_mp3/query')
    for filename in filenames:
        print(filename)
        output= filename.replace('.mp3', '.wav')
        convert_mp3_to_wav(file_path=f'data_mp3/query/{filename}', output= f'data_wav/query/{output}')


    # sample_rate, audio_data = read_audio_by_librosa('data_wav/-4SrLmSxCLg_speech_female_0000_0130.wav')
    # sr, audio = read_audio_by_wavfile('data_wav/-4SrLmSxCLg_speech_female_0000_0130.wav')
    # sr, audio = read_mp3_by_AudioSegment('data_wav/-4SrLmSxCLg_speech_female_0000_0130.wav')
    # print(1)
