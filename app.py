import streamlit as st
import cv2
import numpy as np
from datetime import datetime

# استيراد محمي لـ Mediapipe
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
except AttributeError:
    # حل بديل لو النسخة فيها مشكلة في المسارات
    from mediapipe.python.solutions import face_detection as mp_face

st.set_page_config(page_title="نظام الحضور الذكي | محمد سلامة", layout="centered")

st.title("👤 نظام البصمة الذكي - محمد سلامة")
st.write("خبير EdTech | SAT English Expert")

# واجهة الكاميرا
img_file = st.camera_input("التقط صورة لتسجيل حضورك")

if img_file:
    # معالجة الصورة
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # تشغيل الحساس (Detector)
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        results = detector.process(img_rgb)

        if results.detections:
            st.success("✅ أهلاً بك يا محمد! تم التعرف على الوجه.")
            st.balloons()
            st.info(f"تم تسجيل الحضور: {datetime.now().strftime('%I:%M %p')}")
        else:
            st.warning("⚠️ لم يتم رصد وجه. من فضلك اقترب من الكاميرا وتأكد من الإضاءة.")

st.sidebar.markdown("---")
st.sidebar.write("نظام حضور ذكي خفيف وسريع")
