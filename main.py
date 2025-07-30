import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime, timedelta  # Import timedelta for cooldown

# --- Configuration ---
KNOWN_FACES_DIR = "knownFaces"
ATTENDANCE_LOG_DIR = "attendance_logs"  # Directory to store attendance CSVs
COOLDOWN_SECONDS = 10  # Time in seconds before a person can be logged again

# Ensure attendance log directory exists
import os

os.makedirs(ATTENDANCE_LOG_DIR, exist_ok=True)

# --- Load Known Faces ---
known_face_encodings = []
known_face_names = []

print("Loading known faces...")
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # Only process image files
        name = os.path.splitext(filename)[0]  # Get name from filename (e.g., "smit" from "smit.jpg")
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"Loaded: {name}")
            else:
                print(f"Warning: No face found in {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

if not known_face_encodings:
    print("No known faces loaded. Please add images to the 'knownFaces' directory.")
    exit()

print(f"Successfully loaded {len(known_face_encodings)} known faces.")

# --- Setup for Live Detection ---
video_capture = cv2.VideoCapture(0)

# Variables to store states for logging
# {name: last_logged_datetime}
last_logged_time = {}

# Open CSV file for the current date
current_date_str = datetime.now().strftime("%Y-%m-%d")  # Use YYYY-MM-DD for better sorting
attendance_file_path = os.path.join(ATTENDANCE_LOG_DIR, f"{current_date_str}_attendance.csv")

# Open in 'a+' mode: creates if not exists, appends if exists
# Use 'newline=""' to prevent extra blank rows
csv_file = open(attendance_file_path, "a+", newline="")
lnwriter = csv.writer(csv_file)

# Write header only if the file is new (or empty)
if os.stat(attendance_file_path).st_size == 0:
    lnwriter.writerow(["Timestamp", "Name", "Status"])
    print(f"Created new attendance log: {attendance_file_path}")
else:
    print(f"Appending to existing attendance log: {attendance_file_path}")

print("Starting attendance system... Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame, exiting...")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert BGR (OpenCV) to RGB (face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Process each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"  # Default to Unknown

        # Compare current face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        # Find the best match if any
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:  # Only if there's an actual match (within tolerance)
            name = known_face_names[best_match_index]

            # --- Attendance Logging Logic ---
            current_time = datetime.now()
            # Check for cooldown: Log if not seen before or if cooldown period has passed
            if name not in last_logged_time or \
                    (current_time - last_logged_time[name]) > timedelta(seconds=COOLDOWN_SECONDS):
                timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                lnwriter.writerow([timestamp_str, name, "Present"])
                csv_file.flush()  # Ensure data is written to disk immediately

                print(f"ATTENDANCE LOGGED: {name} at {timestamp_str}")
                last_logged_time[name] = current_time  # Update last logged time for this person

        # --- Drawing on Frame ---
        # Scale back up face locations (as we resized the frame earlier)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Green box for recognized

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow("Attendance System", frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
video_capture.release()
cv2.destroyAllWindows()
csv_file.close()  # Close the CSV file
print("Attendance system stopped.")