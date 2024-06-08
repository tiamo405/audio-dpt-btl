import numpy as np
import librosa
from scipy.io import wavfile
from scipy.fft import fft
def get_100hz_200hz(filepath):
    y, sr = librosa.load(filepath, sr=44800)

    # Tính toán DFT của tín hiệu âm thanh
    D = np.abs(librosa.stft(y))

    # Tần số tương ứng với mỗi hàng của D
    freqs = librosa.fft_frequencies(sr=sr)

    # Tìm chỉ số của các tần số mong muốn
    freq_100hz_idx = np.argmin(np.abs(freqs - 100))
    freq_200hz_idx = np.argmin(np.abs(freqs - 200))

    # Lấy biên độ tại các tần số 100 Hz và 200 Hz
    amplitude_at_100hz = np.max(D[freq_100hz_idx, :])
    amplitude_at_200hz = np.max(D[freq_200hz_idx, :])

    print("Biên độ tại 100 Hz:", amplitude_at_100hz)
    print("Biên độ tại 200 Hz:", amplitude_at_200hz)


    sample_rate, data = wavfile.read(filepath)

    # Thực hiện phân tích phổ Fourier
    fft_output = fft(data)

    # Tính tần số của mỗi điểm trong phổ
    frequencies = np.fft.fftfreq(len(data), 1/sample_rate)

    # Tìm chỉ mục của các tần số mong muốn (100Hz và 200Hz)
    index_100Hz = np.where(np.round(frequencies) == 100)[0][0]
    index_200Hz = np.where(np.round(frequencies) == 200)[0][0]

    # Lấy biên độ ứng với các tần số 100Hz và 200Hz
    amplitude_100Hz = np.abs(fft_output[index_100Hz])
    amplitude_200Hz = np.abs(fft_output[index_200Hz])

    print("Biên độ của tần số 100Hz:", amplitude_100Hz)
    print("Biên độ của tần số 200Hz:", amplitude_200Hz)

    samples, sample_rate = librosa.load(filepath, sr=None)

    # Tính biên độ ứng với tần số 100 Hz và 200 Hz
    frequency_100hz_amplitude = abs(librosa.amplitude_to_db(librosa.stft(samples, n_fft=2048), ref=np.max)[100])
    frequency_200hz_amplitude = abs(librosa.amplitude_to_db(librosa.stft(samples, n_fft=2048), ref=np.max)[200])

    print(f"Biên độ ứng với tần số 100 Hz: {min(frequency_100hz_amplitude)} dB")
    print(f"Biên độ ứng với tần số 200 Hz: {min(frequency_200hz_amplitude)} dB")


get_100hz_200hz('data_wav/gallery/_OkTw766oCs_lecture_male_0247_0323.wav')