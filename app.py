import streamlit as st
import cv2
from ultralytics import YOLO
import pyttsx3

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🎯 VocaSight")
camera_choice = st.radio("Select Camera:", ["Front", "Back"])

# Map choice to OpenCV camera index
# Note: On laptops, usually 0 is default camera. 
# On mobile devices with Streamlit sharing, OpenCV can't directly select front/back camera.
camera_index = 0 if camera_choice == "Front" else 1  # You may need to adjust indexes

# ------------------------------
# Initialize TTS
# ------------------------------
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

speak("Hello! VocaSight is now active.")
st.write(f"{camera_choice} camera is starting...")

# ------------------------------
# Load YOLO model
# ------------------------------
model = YOLO("yolov8n.pt")

# ------------------------------
# Start camera
# ------------------------------
cap = cv2.VideoCapture(camera_index)
FRAME_WINDOW = st.image([])

previous_objects = set()

while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("No camera detected.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)[0]
    current_objects = set()

    for result in results.boxes:
        cls_id = int(result.cls[0])
        conf = float(result.conf[0])
        if conf < 0.5:
            continue
        name = model.names[cls_id]
        current_objects.add(name)

        x1, y1, x2, y2 = map(int, result.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    new_objects = current_objects - previous_objects
    for obj in new_objects:
        speak(f"Obstacle detected: {obj}")

    previous_objects = current_objects
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
