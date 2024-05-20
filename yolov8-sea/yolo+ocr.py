from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import os

def correct_skew(image):
    # 使用Canny边缘检测
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    # 使用Hough变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    print(lines)
    if lines is not None:
        # 计算倾斜角度
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
            angles.append(angle)
        # 计算平均倾斜角度
        median_angle = np.median(angles)
        print(median_angle)
        # 旋转图像以矫正倾斜
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        corrected_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return corrected_img
    return image

# 初始化 YOLO 模型
yolo = YOLO("", task="detect")

# 对图像进行检测
result = yolo(source="chip.jpg", save=True, conf=0.1)

# 初始化 EasyOCR reader
reader = easyocr.Reader(['en'])

# 输出目录
output_dir = "processed_images"
os.makedirs(output_dir, exist_ok=True)

# 检查是否有检测结果
if len(result) > 0 and hasattr(result[0], 'boxes') and hasattr(result[0].boxes, 'xywh'):
    # 加载原始图像
    original_image = cv2.imread("chip.jpg")

    # 提取所有边界框（使用 xywh 格式）
    bboxes = result[0].boxes.xywh.cpu().numpy()

    # 遍历所有检测到的边界框
    for i, bbox in enumerate(bboxes):
        # 提取中心点坐标和宽高
        xc, yc, w, h = bbox

        # 转换为 x1, y1, x2, y2 格式
        x1, y1 = int(xc - w/2), int(yc - h/2)
        x2, y2 = int(xc + w/2), int(yc + h/2)

        # 裁剪对象
        cropped_image = original_image[y1:y2, x1:x2]

        # 放大图像
        resized_image = cv2.resize(cropped_image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # 图像平滑
        smoothed_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

        # 灰度化
        gray_image = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2GRAY)

        # Otsu's 阈值法
        _, otsu_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 矫正文本倾斜
        corrected_image = correct_skew(otsu_thresh)

        # EasyOCR进行OCR检测
        text_otsu = reader.readtext(corrected_image)
        print(f"EasyOCR Result for bbox {i}: {[text[1] for text in text_otsu]}")

        # 保存图像
        cv2.imwrite(os.path.join(output_dir, f"resized_{i}.jpg"), resized_image)
        cv2.imwrite(os.path.join(output_dir, f"smoothed_{i}.jpg"), smoothed_image)
        cv2.imwrite(os.path.join(output_dir, f"gray_{i}.jpg"), gray_image)
        cv2.imwrite(os.path.join(output_dir, f"otsu_thresh_{i}.jpg"), otsu_thresh)
        cv2.imwrite(os.path.join(output_dir, f"corrected_{i}.jpg"), corrected_image)
