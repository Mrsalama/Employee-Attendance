import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import importlib

st.set_page_config(page_title="نظام الحضور الذكي | محمد سلامة", layout="centered")

st.title("👤 نظام البصمة الذكي - محمد سلامة")

# محاولة استيراد المكتبة بأكثر من طريقة
mp_face = None
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
except:
    try:
        import mediapipe.python.solutions.face_detection as mp_face
    except:
        st.error("جاري تهيئة النظام... من فضلك انتظر دقيقة وأعد تحميل الصفحة.")

if mp_face:
    img_file = st.camera_input("التقط صورة لتسجيل حضورك")

    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
            results = detector.process(img_rgb)

            if results.detections:
                st.success("✅ تم تسجيل حضورك بنجاح!")
                st.balloons()
            else:
                st.warning("⚠️ لم يتم رصد وجه. حاول مرة أخرى.")
else:
    st.info("النظام قيد التحميل... تأكد من صحة ملف requirements.txt")
