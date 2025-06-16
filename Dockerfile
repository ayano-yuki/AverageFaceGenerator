FROM python:3.10-slim

# システムパッケージのインストール（dlib用ビルド環境とOpenCV依存）
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libopenblas-dev \
    libx11-dev \
    libgtk-3-dev \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# 必要ファイルをコピー
COPY average_face_delaunay.py .
COPY shape_predictor_68_face_landmarks.dat .

# Pythonパッケージインストール
RUN pip install --no-cache-dir numpy opencv-python dlib scipy

# imagesフォルダはボリュームマウント予定なのでコピーしない

CMD ["python", "average_face_delaunay.py"]
