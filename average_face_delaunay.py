import cv2
import dlib
import numpy as np
import os
from glob import glob
from scipy.spatial import Delaunay

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
IMAGE_DIR = "images"
OUTPUT_SIZE = (600, 600)

# 顔検出器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(image):
    faces = detector(image, 1)
    if len(faces) == 0:
        return None
    return np.array([[p.x, p.y] for p in predictor(image, faces[0]).parts()])

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # 防御チェック：範囲外やサイズ0の矩形を無視
    h, w = img1.shape[:2]
    if (r1[0] < 0 or r1[1] < 0 or r1[0]+r1[2] > w or r1[1]+r1[3] > h or
        r2[0] < 0 or r2[1] < 0 or r2[0]+r2[2] > w or r2[1]+r2[3] > h or
        r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0):
        return

    t1_rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2_rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]

    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    if img1_rect.size == 0:
        return

    size = (r2[2], r2[3])
    warped = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] *= (1 - mask)
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] += warped * mask

def calculate_average_face(images, landmarks_list, output_size=OUTPUT_SIZE):
    average_points = np.mean(np.array(landmarks_list), axis=0)
    subdiv = Delaunay(average_points)
    average_image = np.zeros((output_size[1], output_size[0], 3), dtype=np.float32)

    for i in range(len(images)):
        img = np.float32(images[i])
        img_warped = np.zeros_like(average_image)

        for tri in subdiv.simplices:
            t1 = [landmarks_list[i][j] for j in tri]
            t2 = [average_points[j] for j in tri]
            warp_triangle(img, img_warped, t1, t2)

        average_image += img_warped

    average_image /= len(images)
    return np.uint8(np.clip(average_image, 0, 255))

# メイン処理
images = []
landmarks_list = []
for file in glob(os.path.join(IMAGE_DIR, "*.jpg")):
    img = cv2.imread(file)
    if img is None:
        continue

    lm = get_landmarks(img)
    if lm is None:
        print(f"スキップ（顔検出失敗）: {file}")
        continue

    # 画像とランドマークをスケーリングして統一
    h, w = img.shape[:2]
    resized_img = cv2.resize(img, OUTPUT_SIZE)
    scale_x = OUTPUT_SIZE[0] / w
    scale_y = OUTPUT_SIZE[1] / h
    lm_scaled = np.array([[int(x * scale_x), int(y * scale_y)] for (x, y) in lm])

    images.append(resized_img)
    landmarks_list.append(lm_scaled)

if images:
    avg_face = calculate_average_face(images, landmarks_list)
    cv2.imwrite("average_face_delaunay.jpg", avg_face)
    print("✅ average_face_delaunay.jpg を出力しました")
else:
    print("⚠ 顔が検出できる画像が見つかりませんでした。")
