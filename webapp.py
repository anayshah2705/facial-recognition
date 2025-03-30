import requests
import cv2
import numpy as np
import streamlit as st
from PIL import Image

BACKEND_URL = "http://localhost:5000"

st.title("Face Recognition Attendance System")

option = st.selectbox("Choose an option:", ["Login", "Logout", "Register"])

file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if option == "Register":
        name = st.text_input("Enter your name:")
        if st.button("Register"):
            files = {"image": file}
            data = {"name": name}
            response = requests.post(f"{BACKEND_URL}/register", files=files, data=data)
            st.write(response.json())

    elif option == "Login":
        if st.button("Login"):
            files = {"image": file}
            response = requests.post(f"{BACKEND_URL}/login", files=files)
            st.write(response.json())

    elif option == "Logout":
        if st.button("Logout"):
            files = {"image": file}
            response = requests.post(f"{BACKEND_URL}/logout", files=files)
            st.write(response.json())
