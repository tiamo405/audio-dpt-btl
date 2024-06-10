import numpy as np
from scipy.io import wavfile

import librosa
import matplotlib.pyplot as plt
import utils_audio
import pandas as pd
import os
from scipy.fft import fft






class Feature_audio:
    def __init__(self, sr = 22500):
        self.sr = sr
        self.n_fft = 2048
        self.hop_length = 512


    def calculate_average_energy(self, audio_file, frame_length=2048, hop_length=512): # Năng lượng trung bình
        # Đọc file audio
        audio_data, sr = utils_audio.read_audio_by_librosa(audio_file, self.sr)
        
        #filter zero value
        audio_data = audio_data[audio_data != 0]
        # Tính tong năng lượng của từng bien độ âm thanh
        frame_energy = (audio_data ** 2).sum(axis=0)

        # Tính năng lượng trung bình của toàn bộ đoạn audio
        average_energy = frame_energy/len(audio_data)

        return average_energy

    def compute_sign_change_rate(self, audio_file): # Tốc độ đổi dấu của tín hiệu

        # Đọc file audio
        audio_data, sr = utils_audio.read_audio_by_wavfile(audio_file, self.sr)
        audio_data = audio_data[audio_data != 0]
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
        audio_data = audio_data[audio_data != 0]
        # Xác định ngưỡng amplitude
        max_amplitude = np.max(np.abs(audio_data))
        threshold_amplitude = threshold * max_amplitude
        # Tìm các mẫu có biên độ nhỏ hơn ngưỡng
        below_threshold_indices = np.where(np.abs(audio_data) < threshold_amplitude)[0]
        # Tính thời gian tương ứng với các mẫu có biên độ nhỏ hơn ngưỡng
        time_below_threshold = len(below_threshold_indices) / len(audio_data)

        return time_below_threshold 



    # def spec_cent(self,wav_file):
    #     audio_data, sample_rate = utils_audio.read_audio_by_librosa(wav_file, self.sr)
    #     cent = librosa.feature.spectral_centroid(y=audio_data, sr = sample_rate)[0]
    #     min_val = min(cent)
    #     max_val = max(cent)
    #     scaled_data = [(x - min_val) / (max_val - min_val) for x in cent] #scaled
    #     return np.mean(scaled_data)
        # return np.mean(cent)

    # def spec_bandwidth(self, wav_file):
    #     audio_data, sample_rate = utils_audio.read_audio_by_librosa(wav_file, self.sr)
    #     bw = librosa.feature.spectral_bandwidth(y=audio_data, sr = sample_rate)[0]
    #     min_val = min(bw)
    #     max_val = max(bw)
    #     scaled_data = [(x - min_val) / (max_val - min_val) for x in bw]
    #     return np.mean(scaled_data)

    # def rolloff(self, wav_file):
    #     audio_data, sample_rate = utils_audio.read_audio_by_librosa(wav_file, self.sr)
    #     roll = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
    #     min_val = min(roll)
    #     max_val = max(roll)
    #     scaled_data = [(x - min_val) / (max_val - min_val) for x in roll]
    #     return np.mean(scaled_data)


    def compute_spectral_centroid(self, stft): # Trọng tâm tần số của tín hiệu
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.sr)  # Tần số tương ứng với STFT
        freqs = freqs[:, np.newaxis]  # Thêm một chiều vào freqs
        magnitude_spectrum = np.abs(stft)

        sum_magnitude = np.sum(magnitude_spectrum, axis=0)
        all_id = np.where(sum_magnitude != 0)[0]
        #save all_id of sum_magnitude and magnitude_spectrum
        magnitude_spectrum = magnitude_spectrum[:, all_id]
        sum_magnitude = sum_magnitude[all_id]
        # print(magnitude_spectrum.shape)
        spectral_centroid = np.sum(freqs * magnitude_spectrum, axis=0) / sum_magnitude
        scaled_centroid = (spectral_centroid - np.min(spectral_centroid)) / (np.max(spectral_centroid) - np.min(spectral_centroid))
        return np.mean(scaled_centroid)
 
    def compute_spectral_bandwidth(self, stft, spectral_centroid): # Do rong tần số của tín hiệu
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.sr)
        freqs = freqs[:, np.newaxis]  # Thêm một chiều vào freqs
        magnitude_spectrum = np.abs(stft)

        sum_magnitude = np.sum(magnitude_spectrum, axis=0)
        all_id = np.where(sum_magnitude != 0)[0]
        #save all_id of sum_magnitude and magnitude_spectrum
        magnitude_spectrum = magnitude_spectrum[:, all_id]
        sum_magnitude = sum_magnitude[all_id]

        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude_spectrum, axis=0) / np.sum(magnitude_spectrum, axis=0))
        scaled_bandwidth = (spectral_bandwidth - np.min(spectral_bandwidth)) / (np.max(spectral_bandwidth) - np.min(spectral_bandwidth))
        return np.mean(scaled_bandwidth)

    def compute_spectral_rolloff(self, stft, rolloff_percent=0.85): # Tần số cuối cùng mà tín hiệu có thể đạt được một tỷ lệ năng lượng
        magnitude_spectrum = np.abs(stft)

        sum_magnitude = np.sum(magnitude_spectrum, axis=0)
        all_id = np.where(sum_magnitude != 0)[0]
        #save all_id of sum_magnitude and magnitude_spectrum
        magnitude_spectrum = magnitude_spectrum[:, all_id]
        sum_magnitude = sum_magnitude[all_id]

        power_spectrum = np.square(magnitude_spectrum)
        cumulative_energy = np.cumsum(power_spectrum, axis=0)
        rolloff_threshold = rolloff_percent * cumulative_energy[-1, :]
        rolloff = np.zeros(magnitude_spectrum.shape[1])
        for i in range(magnitude_spectrum.shape[1]):
            rolloff[i] = np.where(cumulative_energy[:, i] >= rolloff_threshold[i])[0][0]
        rolloff_freq = rolloff * (self.sr / magnitude_spectrum.shape[0])
        scaled_rolloff = (rolloff_freq - np.min(rolloff_freq)) / (np.max(rolloff_freq) - np.min(rolloff_freq))
        return np.mean(scaled_rolloff)

    def compute_stft(self, wav_file):
        y, sample_rate = utils_audio.read_audio_by_librosa(wav_file, self.sr)
        n_fft = self.n_fft
        hop_length = self.hop_length
        sr = self.sr
        window = np.hanning(n_fft)
        stft = np.array([fft(window * y[i:i + n_fft])[:n_fft//2+1] for i in range(0, len(y) - n_fft, hop_length)])
        return stft.T


    def all_feature(self, wav_file):
        ave_energy = self.calculate_average_energy(wav_file)
        change_rate = self.compute_sign_change_rate(wav_file)
        silence_ratio = self.calculate_silence_ratio_wav(wav_file)
        stft = self.compute_stft(wav_file)
        # print(stft.shape)
        # print(stft)
        spec_cent = self.compute_spectral_centroid(stft)
        spec_bw = self.compute_spectral_bandwidth(stft, spec_cent)
        roll = self.compute_spectral_rolloff(stft)
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

    # name_audios = os.listdir('data_wav/gallery')
    # for name_audio in name_audios:
    #     print(name_audio)
    #     path_audio = f'data_wav/gallery/{name_audio}'
    #     np_save = f'data_npy/gallery/{name_audio.replace(".wav", ".npy")}'
    #     np.save(np_save, fa.all_feature(path_audio))

    path_audio = 'data_wav/gallery/1RmTc8Vr4Mg_podcast_female_0010_0110.wav'
    ft = fa.all_feature(path_audio)
    print(ft)

