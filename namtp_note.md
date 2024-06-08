# setup
conda create -n audio python=3.7
conda activate audio
pip install -r requirements.txt
pip install torch==1.13.0+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu



# clean.py
- tiền xử lí dữ liệu
    - từ 1 file audio đọc, tách thành 1 kênh ( có audio là 2 kênh gọi là streo gì đó) và chuyển mẫu (samples) từ 44100hz về 16000 Hz
    - sau đó có 1 mảng là chứa các số  ý nghĩa là coi như độ cao
    - cho vào 1 mảng, dùng pandas để biến đổi mảng thành trị cho dương hết, dùng cửa sổ trượt để góm các số liên tiếp vào và biến chúng thnahf gái trị lớn nhất ( kiểu biến 1 cụm các sô ( ví dụ 0 : rate/20) đều bằng số lớn nhất trong đó [0 1 2 3 4 5 6 7 8 9] window = 5 -> [4 4 4 4 4 9 9 9 9 9 ] )
    - sau đó so sánh từng số với threshold để lọc, nêu s, thì cho = false tức = 0
    - sau đó cắt mảng đó ra thành các đoạn auido nhỏ hơn với độ dài đc nhập
    ( ví dụ cắt thành 2s thì lúc biến đổi từ 7s 44100hz có 1 mảng 313992 số  thành 16000 hz là mảng còn 113920 số. sau đó cắt thành các mảng gồm 2*16000 số )

