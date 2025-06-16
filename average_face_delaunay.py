import cv2
import dlib
import numpy as np
import os
from glob import glob
from scipy.spatial import Delaunay

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
IMAGE_DIR = "images"

# 顔検出器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(image):
    faces = detector(image, 1)
    if len(faces) == 0:
        return None
    return np.array([[p.x, p.y] for p in predictor(image, faces[0]).parts()])

def rect_to_bb(rect):
    return (rect.left(), rect.top(), rect.width(), rect.height())

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1_rect = []
    t2_rect = []
    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    size = (r2[2], r2[3])
    warped = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] *= (1 - mask)
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] += warped * mask

def calculate_average_face(images, landmarks_list, output_size=(600, 600)):
    all_points = []
    for lm in landmarks_list:
        all_points.append(np.array(lm))

    average_points = np.mean(np.array(all_points), axis=0)
    rect = (0, 0, output_size[0], output_size[1])
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
        continue
    images.append(cv2.resize(img, (600, 600)))
    landmarks_list.append(lm)

if images:
    avg_face = calculate_average_face(images, landmarks_list)
    cv2.imwrite("average_face_delaunay.jpg", avg_face)
    print("average_face_delaunay.jpg を出力しました")
else:
    print("顔が検出できる画像が見つかりませんでした。")
