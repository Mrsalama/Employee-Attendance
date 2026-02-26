import streamlit as st
import cv2
import numpy as np
from datetime import datetime

# استيراد مباشر وبسيط
try:
    import mediapipe as mp
except ImportError:
    st.error("المكتبات لا تزال في مرحلة التثبيت... برجاء الانتظار 30 ثانية ثم تحديث الصفحة (Refresh).")
    st.stop()

st.set_page_config(page_title="نظام الحضور الذكي | محمد سلامة", layout="centered")

st.title("👤 نظام البصمة الذكي - محمد سلامة")
st.write("SAT English Expert & EdTech Developer")

# تشغيل أدوات جوجل
mp_face = mp.solutions.face_detection

img_file = st.camera_input("التقط صورة لتسجيل حضورك")

if img_file:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        results = detector.process(img_rgb)

        if results.detections:
            st.success("✅ تم تسجيل حضورك بنجاح يا مستر محمد!")
            st.balloons()
            st.info(f"الوقت الحالي: {datetime.now().strftime('%I:%M %p')}")
        else:
            st.warning("⚠️ لم يتم رصد وجه واضح. حاول مرة أخرى مع إضاءة أفضل.")
