import librosa
import soundfile as sf

# Đường dẫn đến file âm thanh ban đầu
audio_file_path = "file.mp3"

# Đọc file âm thanh và lấy mẫu số (sampling rate) của nó
audio_data, sr = librosa.load(audio_file_path, sr=None)

# Xác định tên và định dạng của file mới
output_file_path = "file_out.mp3"

# Lưu dữ liệu âm thanh sang file mới
sf.write(output_file_path, audio_data, sr)
