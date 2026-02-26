import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

# إعدادات الصفحة
st.set_page_config(page_title="نظام الحضور الذكي | محمد سلامة", layout="centered")

st.title("👤 نظام البصمة الذكي أونلاين")
st.markdown("---")

# استدعاء حلول جوجل بطريقة متوافقة مع بايثون 3.13
face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

img_file = st.camera_input("التقط صورة لتسجيل حضورك")

if img_file:
    # معالجة الصورة
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # تشغيل التعرف على الوجه
    with face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        results = detector.process(img_rgb)

        if results.detections:
            st.success("✅ تم التعرف على الوجه بنجاح!")
            now = datetime.now().strftime("%I:%M:%S %p")
            st.info(f"مرحباً بك يا أستاذ محمد. تم تسجيل الحضور الساعة: {now}")
            st.balloons()
        else:
            st.error("❌ لم يتم رصد وجه واضح. حاول ضبط الإضاءة والوقوف أمام الكاميرا مباشرة.")

st.sidebar.info("هذا النظام يعمل بتقنية Google MediaPipe")
