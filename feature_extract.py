import numpy as np
from scipy.io import wavfile

import librosa
import matplotlib.pyplot as plt
import utils_audio
import pandas as pd
import os

class Feature_audio:
    def __init__(self, sr = 22500):
        self.sr = sr
    

    def calculate_average_energy(self, audio_file, frame_length=2048, hop_length=512): # Năng lượng trung bình
        # Đọc file audio
        audio_data, sr = utils_audio.read_audio_by_librosa(audio_file, self.sr)


        # Tính tong năng lượng của từng bien độ âm thanh
        frame_energy = (audio_data ** 2).sum(axis=0)

        # Tính năng lượng trung bình của toàn bộ đoạn audio
        average_energy = frame_energy/len(audio_data)

        return average_energy

    def compute_sign_change_rate(self, audio_file): # Tốc độ đổi dấu của tín hiệu

        # Đọc file audio
        audio_data, sr = utils_audio.read_audio_by_wavfile(audio_file, self.sr)

        # test
        # audio_data = np.array([1, -2, 3, -4, -1, -1])
        # Chuyển đổi tín hiệu thành dấu (+1, 0, -1)
        signs = np.sign(audio_data)
        
        # Tính toán số lượng lần đổi dấu
        sign_changes = np.sum(np.abs(np.diff(signs)))
        
        # Tính tốc độ đổi dấu bằng cách chia số lượng lần đổi dấu cho độ dài của tín hiệu
        sign_change_rate = sign_changes / (2*len(audio_data))
        
        return sign_change_rate

    def calculate_silence_ratio_wav(self, wav_file, threshold=0.01): # Tỉ lệ của khoảng lặng trong âm
        audio_data, sample_rate = utils_audio.read_audio_by_wavfile(wav_file, self.sr)
        # audio_data = np.array([100, -200, 150, 50, -120, 180, -90])
        
        # Xác định ngưỡng amplitude
        max_amplitude = np.max(np.abs(audio_data))
        threshold_amplitude = threshold * max_amplitude
        # Tìm các mẫu có biên độ nhỏ hơn ngưỡng
        below_threshold_indices = np.where(np.abs(audio_data) < threshold_amplitude)[0]
        # Tính thời gian tương ứng với các mẫu có biên độ nhỏ hơn ngưỡng
        time_below_threshold = len(below_threshold_indices) / sample_rate

        return time_below_threshold/100



    def spec_cent(self,wav_file):
        audio_data, sample_rate = utils_audio.read_audio_by_librosa(wav_file, self.sr)
        cent = librosa.feature.spectral_centroid(y=audio_data, sr = sample_rate)[0]
        min_val = min(cent)
        max_val = max(cent)
        scaled_data = [(x - min_val) / (max_val - min_val) for x in cent]
        return np.mean(scaled_data)
        # return np.mean(cent)

    def spec_bandwidth(self, wav_file):
        audio_data, sample_rate = utils_audio.read_audio_by_librosa(wav_file, self.sr)
        bw = librosa.feature.spectral_bandwidth(y=audio_data, sr = sample_rate)[0]
        min_val = min(bw)
        max_val = max(bw)
        scaled_data = [(x - min_val) / (max_val - min_val) for x in bw]
        return np.mean(scaled_data)

    def rolloff(self, wav_file):
        audio_data, sample_rate = utils_audio.read_audio_by_librosa(wav_file, self.sr)
        roll = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        min_val = min(roll)
        max_val = max(roll)
        scaled_data = [(x - min_val) / (max_val - min_val) for x in roll]
        return np.mean(scaled_data)


    def all_feature(self, wav_file):
        ave_energy = self.calculate_average_energy(wav_file)
        change_rate = self.compute_sign_change_rate(wav_file)
        silence_ratio = self.calculate_silence_ratio_wav(wav_file)
        spec_cent = self.spec_cent(wav_file)
        spec_bw = self.spec_bandwidth(wav_file)
        roll = self.rolloff(wav_file)
        vector = [ave_energy, change_rate, silence_ratio, spec_cent, spec_bw, roll]
        return np.array(vector)


    def plot_audio_spectrum(self, file_path):
        # Đọc file WAV
        sample_rate, audio_data = utils_audio.read_audio_by_wavfile(file_path)
        
        # Tính toán phổ âm của dữ liệu âm thanh
        spectrum = np.fft.fft(audio_data)
        frequencies = np.fft.fftfreq(len(audio_data), d=1/sample_rate)
        amplitude_spectrum = np.abs(spectrum)
        
        # Thể hiện phổ âm bằng biểu đồ
        plt.figure(figsize=(10, 4))
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(audio_data)) / sample_rate, audio_data)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Audio Waveform')
        
        plt.subplot(2, 1, 2)
        plt.plot(frequencies, amplitude_spectrum)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude Spectrum')
        plt.title('Audio Spectrum')
        plt.xlim(0, sample_rate / 2)  # Chỉ hiển thị phổ âm cho tần số dương
        plt.tight_layout()
        plt.savefig('d.png')


if __name__ == '__main__':

    fa = Feature_audio(sr=44100)
    name_audios = os.listdir('data_wav/query')
    for name_audio in name_audios:
        print(name_audio)
        path_audio = f'data_wav/query/{name_audio}'
        np_save = f'data_npy/query/{name_audio.replace(".wav", ".npy")}'
        np.save(np_save, fa.all_feature(path_audio))

