import streamlit as st
import cv2
import numpy as np
from datetime import datetime

# استيراد استثنائي لحل مشكلة AttributeError في النسخ الجديدة
try:
    import mediapipe as mp
    # محاولة الاستيراد المباشر من المسار الداخلي
    from mediapipe.python.solutions import face_detection as mp_face
except:
    try:
        import mediapipe.solutions.face_detection as mp_face
    except Exception as e:
        st.error("نعتذر، النظام لا يزال يربط المكتبات. برجاء الضغط على Reboot من القائمة.")
        st.stop()

st.set_page_config(page_title="نظام الحضور الذكي | محمد سلامة", layout="centered")

st.title("👤 نظام البصمة الذكي - محمد سلامة")
st.write("SAT English Expert & EdTech Developer")

# واجهة الكاميرا
img_file = st.camera_input("التقط صورة لتسجيل حضورك")

if img_file:
    # معالجة الصورة
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # تشغيل التعرف على الوجه
        with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
            results = detector.process(img_rgb)

            if results.detections:
                st.success("✅ أهلاً بك! تم التعرف على الوجه بنجاح.")
                st.balloons()
                st.info(f"وقت تسجيل الحضور: {datetime.now().strftime('%I:%M %p')}")
            else:
                st.warning("⚠️ لم يتم رصد وجه. حاول الاقتراب من الكاميرا.")
