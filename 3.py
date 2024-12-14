import streamlit as st
import cv2
from keras.models import model_from_json
import numpy as np
from tensorflow.keras.models import load_model
from keras import backend as K

import pathlib

st.set_page_config(
    page_title="Emotion Detector",  # Title for the browser tab
    page_icon="🎥",  # Icon for the browser tab
    layout="centered",  # Centers the content on the page
    initial_sidebar_state="collapsed"  # Optional: hide the sidebar initially
)

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
css_path = pathlib.Path("assets/style.css")
load_css(css_path)

def clear_session():
    K.clear_session()  
    print("TensorFlow session cleared.")


model = load_model("emotiondetector.h5")


haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


def run_app():
    stframe = st.empty()
    cap = st.session_state.get("camera", None)
    
    if not cap or not cap.isOpened():
        st.error("Unable to access the camera.")
        st.session_state.run = False
        return

    try:
        while st.session_state.get("run", False):
            ret, im = cap.read()
            if not ret:
                st.warning("Unable to read from the webcam. Stopping...")
                break

          
            img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(img, 1.3, 5)

           
            for (x, y, w, h) in faces:
                face = img[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                face_features = extract_features(face)
                prediction = model.predict(face_features)
                label = labels[prediction.argmax()]
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(im, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

     
            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            stframe.image(im_rgb, channels="RGB")
    finally:
      
        if cap:
            cap.release()
            st.session_state.camera = None
        clear_session() 

if "run" not in st.session_state:
    st.session_state.run = False
if "start_disabled" not in st.session_state:
    st.session_state.start_disabled = False
if "camera" not in st.session_state:
    st.session_state.camera = None


st.markdown('<h1 class="blue_gradient">Emotion Detector</h1>', unsafe_allow_html=True)


    
col1, col2 = st.columns(2)

with col1:
    if st.button("Start Camera",key="green"):
        st.session_state.start_disabled = True
        st.session_state.run = True
        with st.spinner("Loading camera..."):
            st.session_state.camera = cv2.VideoCapture(0)
        

with col2:
    if st.button("Stop Camera",key="red"):
        st.session_state.run = False
        st.session_state.start_disabled = False
        if st.session_state.camera:
            st.session_state.camera.release()
            st.session_state.camera = None
        clear_session()  

if not st.session_state.run:
    st.markdown(
        """
        <div class="camera-placeholder">
            <i class="fa fa-video-camera-slash"></i>
            Camera is off
        </div>
        """, unsafe_allow_html=True)

else:
    run_app()