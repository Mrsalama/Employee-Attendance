import streamlit as st
import face_recognition
import sqlite3
import pandas as pd
import pickle
from datetime import datetime
import numpy as np

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="نظام الحضور الذكي | محمد سلامة",
    page_icon="👤",
    layout="wide"
)

# --- دالة إنشاء قاعدة البيانات ---
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    # جدول الموظفين (الاسم وبصمة الوجه)
    c.execute('CREATE TABLE IF NOT EXISTS employees (name TEXT, encoding BLOB)')
    # جدول سجل الحضور
    c.execute('CREATE TABLE IF NOT EXISTS logs (name TEXT, type TEXT, time TEXT)')
    conn.commit()
    conn.close()

init_db()

# --- واجهة المستخدم ---
st.title("🚀 نظام الحضور والانصراف بالذكاء الاصطناعي")
st.markdown(f"**المطور:** محمد سلامة | خبير EdTech")

# القائمة الجانبية
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
menu = ["🏠 تسجيل الحضور", "➕ إضافة موظف جديد", "📊 سجل التقارير"]
choice = st.sidebar.selectbox("اختر من القائمة:", menu)

# --- القسم الأول: تسجيل الحضور ---
if choice == "🏠 تسجيل الحضور":
    st.subheader("📸 التحقق من الهوية عبر الكاميرا")
    img_file = st.camera_input("التقط صورة للتحقق")
    
    if img_file:
        with st.spinner("جاري معالجة الصورة والتعرف على الوجه..."):
            # تحويل الصورة الملتقطة إلى تنسيق يفهمه face_recognition
            image = face_recognition.load_image_file(img_file)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                user_enc = encodings[0]
                conn = sqlite3.connect('attendance.db')
                c = conn.cursor()
                c.execute("SELECT name, encoding FROM employees")
                all_employees = c.fetchall()
                
                found = False
                for name, stored_bytes in all_employees:
                    stored_face = pickle.loads(stored_bytes)
                    # مقارنة الوجه الحالي بالوجوه المخزنة
                    matches = face_recognition.compare_faces([stored_face], user_enc, tolerance=0.6)
                    
                    if matches[0]:
                        st.success(f"✅ أهلاً بك يا {name}! تم تسجيل حضورك بنجاح.")
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        c.execute("INSERT INTO logs VALUES (?, ?, ?)", (name, "بصمة وجه", now))
                        conn.commit()
                        st.balloons()
                        found = True
                        break
                
                if not found:
                    st.error("❌ عذراً، لم يتم العثور على اسمك في قاعدة البيانات.")
                conn.close()
            else:
                st.warning("⚠️ لم يتم رصد وجه واضح. تأكد من الإضاءة ووجه الكاميرا جيداً.")

# --- القسم الثاني: إضافة موظف جديد ---
elif choice == "➕ إضافة موظف جديد":
    st.subheader("📝 تسجيل موظف جديد في النظام")
    new_name = st.text_input("اسم الموظف الثلاثي")
    new_img = st.camera_input("التقط الصورة المرجعية (بصمة الوجه)")
    
    if st.button("حفظ البيانات") and new_name and new_img:
        with st.spinner("جاري حفظ البصمة..."):
            image = face_recognition.load_image_file(new_img)
            encs = face_recognition.face_encodings(image)
            if encs:
                enc = encs[0]
                conn = sqlite3.connect('attendance.db')
                c = conn.cursor()
                # تخزين مصفوفة الوجه بعد تحويلها لـ Bytes
                c.execute("INSERT INTO employees VALUES (?, ?)", (new_name, pickle.dumps(enc)))
                conn.commit()
                conn.close()
                st.success(f"✅ تم تسجيل الموظف '{new_name}' بنجاح في النظام.")
            else:
                st.error("❌ فشل النظام في التقاط ملامح الوجه. حاول مرة أخرى.")

# --- القسم الثالث: سجل التقارير ---
elif choice == "📊 سجل التقارير":
    st.subheader("📅 تقارير الحضور والانصراف")
    conn = sqlite3.connect('attendance.db')
    df = pd.read_sql_query("SELECT * FROM logs", conn)
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        # إمكانية تحميل التقرير كملف CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 تحميل التقرير كملف Excel/CSV", data=csv, file_name="attendance_report.csv", mime="text/csv")
    else:
        st.info("لا توجد سجلات حضور حتى الآن.")
    conn.close()

# --- تذييل الصفحة ---
st.sidebar.markdown("---")
st.sidebar.write("💻 **EdTech Innovation**")
st.sidebar.write("By: Muhammad Salama")