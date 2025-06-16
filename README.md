# Average Face Generator with Python + Docker
このプロジェクトは、**複数の顔写真から「平均顔」画像を生成する**ツールです。Dlibによる顔検出・ランドマーク検出、およびDelaunay三角分割＋アフィン変換を用いた自然な平均合成を行います。また、環境構築は、Dockerコンテナを用いて行います。

## 📦 構成
```sh
./
├── images/                                 # 入力画像（正面の顔写真）
│ ├── 1.jpg
│ ├── 2.jpg
├── average_face_delaunay.py                # メインスクリプト
├── shape_predictor_68_face_landmarks.dat   # dlibの顔ランドマークモデル
├── Dockerfile                              # Docker設定
```

## ✅ 必要なもの
### 1. 顔ランドマークモデル

以下のdlibモデルをダウンロードして、このプロジェクト直下に配置してください：
🔗 [Download shape_predictor_68_face_landmarks.dat](https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat)

### 2. 入力画像を `images/` に入れる
- 正面から撮影された人物の顔写真（JPEG）を数枚以上
- 解像度は 600x600px 前後がおすすめ（スクリプト内でリサイズされます）

## 🚀 実行方法（PowerShell）
### 1. Dockerイメージをビルド
```powershell
docker build -t average-face .
```

### 2. 実行（画像フォルダをマウント）
```powershell
docker run --rm -v ${PWD}\images:/app/images -v ${PWD}:/app average-face
```
| ※ ${PWD} は現在のディレクトリを表すPowerShellの変数です

## 🖼️ 出力
- 成功すると average_face_delaunay.jpg が生成されます
- スクリプトと同じ場所に保存されます

## 🧠 技術的な背景
- 顔検出：dlib.get_frontal_face_detector()
- ランドマーク抽出：shape_predictor_68_face_landmarks.dat（68点）
- 平均顔構築：
    - 顔の各ランドマーク位置をアライン
    - Delaunay三角分割で顔を三角形に分割
    - 各三角形をアフィン変換で合成
    - 最終的にすべての顔を平均化