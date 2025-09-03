import cv2
import face_recognition
import pickle
import numpy as np
import time
import simpleaudio as sa  

# Load precomputed model
with open("face_model.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# Load alarm sound
wave_obj = sa.WaveObject.from_wave_file("alarm.wav")  

video = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit")

face_locations = []
face_names = []
last_process_time = 0
process_interval = 0.5  

unlock_start_time = None
unlock_duration = 3.0  
door_status = "Locked"

alarm_start_time = None
alarm_duration = 3.0
alarm_playing = False
show_unknown_text = False

# State variable for red light
red_light_active = False

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    current_time = time.time()
    unknown_detected_this_frame = False

    # Process face every interval
    if current_time - last_process_time > process_interval:
        last_process_time = current_time

        face_locations_small = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings_small = face_recognition.face_encodings(rgb_small_frame, face_locations_small)

        face_names = []
        face_locations = []

        for (top, right, bottom, left), face_encoding in zip(face_locations_small, face_encodings_small):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            name = "Unknown"
            accuracy = 0.0

            if True in matches:
                best_match_index = np.argmin(face_distances)
                name = known_names[best_match_index]
                accuracy = (1 - face_distances[best_match_index]) * 100
                unlock_start_time = current_time
            else:
                name = "Unknown - Not registered"
                unknown_detected_this_frame = True
                # Start alarm timer
                alarm_start_time = current_time
                show_unknown_text = True
                if not alarm_playing:
                    play_obj = wave_obj.play()
                    alarm_playing = True

            face_names.append(f"{name} ({accuracy:.2f}%)")
            face_locations.append((top*4, right*4, bottom*4, left*4))

    # Countdown for door unlock
    if unlock_start_time is not None:
        remaining = int(unlock_duration - (current_time - unlock_start_time) + 1)
        if remaining > 0:
            door_status = f"Door Unlocked: {remaining} sec"
        else:
            door_status = "Locked"
            unlock_start_time = None

    # Countdown for unknown alarm text
    if show_unknown_text:
        elapsed_alarm = current_time - alarm_start_time
        if elapsed_alarm > alarm_duration:
            show_unknown_text = False
            alarm_playing = False

    # Set red light state
    red_light_active = show_unknown_text or unknown_detected_this_frame

    # Draw rectangles and labels
    for ((top, right, bottom, left), label) in zip(face_locations, face_names):
        color = (0, 255, 0) if "Unknown" not in label else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Door status top-left
    cv2.putText(frame, f"Door Status: {door_status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Unknown text in center
    if show_unknown_text:
        text = "UNKNOWN - NOT REGISTERED"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
        cv2.putText(frame, text, ((frame.shape[1]-w)//2, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Status light (red/green) top-right
    light_color = (0, 0, 255) if red_light_active else (0, 255, 0)
    cv2.circle(frame, (frame.shape[1]-30, 30), 15, light_color, -1)

    cv2.imshow("Face Recognizer - Door Lock Simulation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
