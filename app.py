import os
import cv2
import numpy as np
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import face_recognition
import pickle
from PIL import Image

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

st.title("Facial Recognition Attendance System")

# Initialize webcam
cap = cv2.VideoCapture(0)


def capture_frame():
    """Capture an image from the webcam"""
    ret, frame = cap.read()
    if ret:
        return frame
    return None


def recognize_user(frame):
    """Recognize user by comparing against stored embeddings"""
    embeddings_unknown = face_recognition.face_encodings(frame)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found'
    else:
        embeddings_unknown = embeddings_unknown[0]

    users_ref = db.collection("users").stream()

    for user in users_ref:
        data = user.to_dict()
        stored_embedding = pickle.loads(data["embedding"])

        match = face_recognition.compare_faces([stored_embedding], embeddings_unknown)[0]
        if match:
            return data["name"]

    return 'unknown_person'


# Authentication Section
st.header("User Authentication")

if st.button("Login"):
    frame = capture_frame()
    if frame is not None:
        user = recognize_user(frame)
        if user in ['unknown_person', 'no_persons_found']:
            st.error("Unknown user. Please register.")
        else:
            st.success(f"Welcome back, {user}!")
            db.collection("attendance").add({
                "name": user,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "status": "in"
            })

if st.button("Logout"):
    frame = capture_frame()
    if frame is not None:
        user = recognize_user(frame)
        if user in ['unknown_person', 'no_persons_found']:
            st.error("Unknown user. Please register.")
        else:
            st.success(f"Goodbye, {user}!")
            db.collection("attendance").add({
                "name": user,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "status": "out"
            })

# New User Registration
st.header("Register New User")
username = st.text_input("Enter your name:")
if st.button("Capture & Register"):
    frame = capture_frame()
    if frame is not None:
        embeddings = face_recognition.face_encodings(frame)
        if embeddings:
            user_data = {
                "name": username,
                "embedding": pickle.dumps(embeddings[0])  # Convert to bytes before storing
            }
            db.collection("users").add(user_data)
            st.success(f"User {username} registered successfully!")
        else:
            st.error("No face detected. Try again.")

# Close webcam when app stops
if st.button("Close Webcam"):
    cap.release()
    st.write("Webcam closed.")
