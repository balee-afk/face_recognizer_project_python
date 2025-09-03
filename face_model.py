import face_recognition
import os
import pickle

DATASET_DIR = "dataset"
MODEL_FILE = "face_model.pkl"  # simpan encoding semua orang

known_encodings = []
known_names = []

for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue

    for filename in os.listdir(person_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(person_folder, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
                print(f"[INFO] Loaded: {person_name} ({filename})")

# Simpan model ke file
with open(MODEL_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print("[INFO] Model saved!")
