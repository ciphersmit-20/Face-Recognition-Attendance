import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
from datetime import datetime, timedelta
import numpy as np  # for face encodings
import csv
import os  # for Directory management
import face_recognition

# --- Configuration ---
KNOWN_FACES_DIR = "knownFaces"
ATTENDANCE_LOG_DIR = "attendance_logs"
COOLDOWN_SECONDS = 10

# Ensure attendance log directory exists
os.makedirs(ATTENDANCE_LOG_DIR, exist_ok=True)


class VideoStreamApp:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # --- THIS LINE WAS MISSING AND IS NOW ADDED ---
        self.is_camera_on = False  # Initialize the camera state flag

        # ___Tkinter UI setup
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.btn_start_stop = tk.Button(window, text="Start Camera", width=50, command=self.toggle_camera)
        self.btn_start_stop.pack(anchor=tk.CENTER, expand=True)

        self.delay = 15

        # ----Attendance system Initialization
        self.known_face_encodings = []
        self.known_face_names = []
        self.last_logged_time = {}  # {name:datetime_object} for cooldown

        self.load_known_faces()  # load Faces from KNOWN_FACES_DIR

        # Changed date format to YYYY-MM-DD for better sorting of filenames
        self.attendance_file_path = os.path.join(ATTENDANCE_LOG_DIR,
                                                 f"{datetime.now().strftime('%Y-%m-%d')}_attendance.csv")
        self.csv_file = open(self.attendance_file_path, "a+", newline="")
        self.lnwriter = csv.writer(self.csv_file)

        if os.stat(self.attendance_file_path).st_size == 0:
            self.lnwriter.writerow(["Timestamp", "Name", "Status"])
            print(f"Created new Attendance log: {self.attendance_file_path}")
        else:
            print(f"Appending to existing attendance log: {self.attendance_file_path}")

        self.update_video_feed()  # Start the loop for updating frames

        self.window.mainloop()  # Tkinter event loop start

    def load_known_faces(self):
        print("Loading known faces...")  # Corrected "Loadding"
        if not os.path.exists(KNOWN_FACES_DIR):
            # Corrected "found" to "NOT found" in error message
            print(f"Error: {KNOWN_FACES_DIR} directory NOT found. Please create it and add face images.")
            return

        for filename in os.listdir(KNOWN_FACES_DIR):
            # Corrected .png extension
            if filename.endswith((".jpg", ".jpeg", ".png")):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(KNOWN_FACES_DIR, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(name)
                        print(f"Loaded: {name}")
                    else:
                        print(f"Warning: No face found in {filename}")

                except Exception as e:
                    print(f"Error loading {filename}: {e}")  # Corrected f-string for error message

        if not self.known_face_encodings:
            print("No known faces loaded. Please add images to the 'knownFaces' directory.")
        else:
            print(f"Successfully loaded {len(self.known_face_encodings)} known faces.")

    def toggle_camera(self):
        if self.is_camera_on:
            if hasattr(self, 'vid') and self.vid.isOpened():
                self.vid.release()
            self.is_camera_on = False
            self.canvas.delete("all")
            self.canvas.create_text(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
                                    text="Camera Off", fill="red", font=("Arial", 24))
            self.btn_start_stop.config(text="Start Camera")
            print("Camera is OFF")
        else:
            self.vid = cv2.VideoCapture(self.video_source)
            if not self.vid.isOpened():
                self.is_camera_on = False
                self.btn_start_stop.config(text="Start Camera")
                self.canvas.delete("all")
                self.canvas.create_text(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
                                        text="Camera Error!", fill="red", font=("Arial", 24))
                print("Error: Camera could not be opened.")
            else:
                self.is_camera_on = True
                self.btn_start_stop.config(text="Stop Camera")
                print("Camera is ON")

    def update_video_feed(self):
        if self.is_camera_on and hasattr(self, 'vid') and self.vid.isOpened():
            ret, frame = self.vid.read()

            # Corrected logic: if not ret (frame read failed)
            if not ret:
                print("Error: Could not read frame from camera. Turning camera off.")
                self.toggle_camera()
            else:
                # --- Face Recognition & logging Logic ---
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    name = "Unknown"  # Corrected "Unkown"

                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]

                        # --- Cooldown Logic ---
                        current_time = datetime.now()
                        # Corrected cooldown logic syntax
                        if name not in self.last_logged_time or \
                                (current_time - self.last_logged_time[name]) > timedelta(seconds=COOLDOWN_SECONDS):
                            timestamp_str = current_time.strftime(
                                "%Y-%m-%d %H:%M:%S")  # Changed to YYYY-MM-DD for consistency
                            self.lnwriter.writerow([timestamp_str, name, "Present"])
                            self.csv_file.flush()

                            print(f"ATTENDANCE LOGGED: {name} at {timestamp_str}")
                            self.last_logged_time[name] = current_time

                    # --- Drawing on frame ---
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Green box
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255),
                                  cv2.FILLED)  # Blue background for text

                    font_type = cv2.FONT_HERSHEY_DUPLEX  # Define font type
                    # Corrected: Use font_type as the font parameter
                    cv2.putText(frame, name, (left + 6, bottom - 6), font_type, 0.8, (255, 255, 255), 1)

                # --- Display the modified frame on Tkinter Canvas ---
                # Corrected cv2Color to cv2.cvtColor
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        else:
            pass  # No changes needed here

        self.window.after(self.delay, self.update_video_feed)

    def __del__(self):
        if hasattr(self, 'vid') and self.vid.isOpened():
            self.vid.release()
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()
        print("Application closing. Camera and CSV file released/closed.")


if __name__ == '__main__':
    root = tk.Tk()
    app = VideoStreamApp(root, "Face Recognition Attendance System")