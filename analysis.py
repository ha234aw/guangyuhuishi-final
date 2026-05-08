from __future__ import annotations

import io
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image


def decode_uploaded_image(raw_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    arr = np.array(image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _load_cascade(name: str) -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + name
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"无法加载 OpenCV 级联分类器：{name}")
    return cascade


FACE_CASCADE = _load_cascade("haarcascade_frontalface_default.xml")
EYE_CASCADE = _load_cascade("haarcascade_eye.xml")


def analyze_sensor_image(image_bgr: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    face_detected = len(faces) > 0
    eye_detected = False
    stability = 25.0
    attention = 25.0

    if face_detected:
        x, y, fw, fh = sorted(faces, key=lambda item: item[2] * item[3], reverse=True)[0]
        face_center_x = x + fw / 2
        frame_center_x = w / 2
        center_offset = abs(face_center_x - frame_center_x) / max(frame_center_x, 1)
        face_scale = min((fw * fh) / max((w * h), 1), 0.3) / 0.3
        stability = max(0.0, min(100.0, 100 * (0.6 * (1 - center_offset) + 0.4 * face_scale)))

        roi_gray = gray[y:y + fh, x:x + fw]
        eyes = EYE_CASCADE.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
        eye_detected = len(eyes) >= 2
        attention = 85.0 if eye_detected else 45.0

    brightness = float(np.mean(gray))
    brightness_score = max(0.0, min(100.0, (brightness / 255.0) * 100.0))

    high_light_mask = gray > 230
    glare_ratio = float(np.mean(high_light_mask)) * 100.0
    glare_risk = max(0.0, min(100.0, glare_ratio * 3.0))

    fatigue = max(0.0, min(100.0, 100 - (0.45 * stability + 0.35 * attention + 0.20 * brightness_score)))
    sensor_score = max(0.0, min(100.0, 0.35 * stability + 0.30 * attention + 0.20 * brightness_score + 0.15 * (100 - glare_risk)))

    advice_parts = []
    if not face_detected:
        advice_parts.append("未稳定检测到人脸，建议正对镜头并保持上半身相对静止。")
    else:
        if stability < 60:
            advice_parts.append("面部稳定度一般，建议调整坐姿与设备角度。")
        if not eye_detected:
            advice_parts.append("双眼检测不够稳定，可能受到反光、遮挡或拍摄角度影响。")
    if brightness_score < 40:
        advice_parts.append("环境亮度偏低，建议在更均匀明亮的条件下完成训练。")
    if glare_risk > 40:
        advice_parts.append("图像中反光风险偏高，建议避免镜片反光或强光直射。")
    if fatigue > 60:
        advice_parts.append("当前疲劳指数偏高，建议缩短本次训练时长并增加休息。")
    if not advice_parts:
        advice_parts.append("当前使用状态较稳定，可继续完成常规训练。")

    return {
        "face_detected": bool(face_detected),
        "eye_detected": bool(eye_detected),
        "brightness": round(brightness_score, 1),
        "stability": round(stability, 1),
        "attention": round(attention, 1),
        "fatigue": round(fatigue, 1),
        "glare_risk": round(glare_risk, 1),
        "sensor_score": round(sensor_score, 1),
        "advice": " ".join(advice_parts),
    }
