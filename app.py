import os
import pickle
import datetime
import json
import cv2
import numpy as np
import face_recognition
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore, storage

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'your-project-id.appspot.com'})
db = firestore.client()
bucket = storage.bucket()

app = Flask(__name__)


def recognize_face(unknown_image):
    embeddings_unknown = face_recognition.face_encodings(unknown_image)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found'

    embeddings_unknown = embeddings_unknown[0]

    # Fetch all registered users
    users_ref = db.collection("users").stream()

    for user in users_ref:
        data = user.to_dict()
        known_embedding = np.array(json.loads(data["embedding"]))

        match = face_recognition.compare_faces([known_embedding], embeddings_unknown)[0]
        if match:
            return data["name"]

    return 'unknown_person'


@app.route('/register', methods=['POST'])
def register():
    file = request.files['image']
    name = request.form['name']

    image = face_recognition.load_image_file(file)
    embeddings = face_recognition.face_encodings(image)

    if len(embeddings) == 0:
        return jsonify({"error": "No face found"}), 400

    # Save face embedding
    db.collection("users").document(name).set({
        "name": name,
        "embedding": json.dumps(embeddings[0].tolist())
    })

    return jsonify({"message": "User registered successfully!"})


@app.route('/login', methods=['POST'])
def login():
    file = request.files['image']
    image = face_recognition.load_image_file(file)

    name = recognize_face(image)

    if name in ['unknown_person', 'no_persons_found']:
        return jsonify({"error": "Face not recognized"}), 401

    db.collection("attendance").add({
        "name": name,
        "time": datetime.datetime.now().isoformat(),
        "status": "in"
    })

    return jsonify({"message": f"Welcome, {name}!"})


@app.route('/logout', methods=['POST'])
def logout():
    file = request.files['image']
    image = face_recognition.load_image_file(file)

    name = recognize_face(image)

    if name in ['unknown_person', 'no_persons_found']:
        return jsonify({"error": "Face not recognized"}), 401

    db.collection("attendance").add({
        "name": name,
        "time": datetime.datetime.now().isoformat(),
        "status": "out"
    })

    return jsonify({"message": f"Goodbye, {name}!"})


if __name__ == '__main__':
    app.run(debug=True)
