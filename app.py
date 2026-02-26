import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime

# إعدادات جوجل لتعرف الوجوه
mp_face_detection = mp.solutions.face_detection
st.set_page_config(page_title="نظام الحضور الذكي | محمد سلامة", layout="wide")

st.title("👤 نظام البصمة الذكي (نسخة السحابة السريعة)")
st.subheader("المطور: محمد سلامة - خبير EdTech")

# واجهة تسجيل الحضور
img_file = st.camera_input("التقط صورة لتسجيل حضورك")

if img_file:
    # تحويل الصورة لمعالجة جوجل
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))

        if results.detections:
            st.success("✅ تم رصد الوجه بنجاح!")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.write(f"تم تسجيل الحضور في: {now}")
            st.balloons()
            
            # عرض بيانات وهمية للسجل (للتجربة)
            df = pd.DataFrame({"الاسم": ["محمد سلامة"], "الوقت": [now], "الحالة": ["حاضر"]})
            st.table(df)
        else:
            st.error("❌ لم يتم رصد وجه واضح، حاول مرة أخرى.")
