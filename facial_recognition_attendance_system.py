import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Open the video cam
video_capture = cv2.VideoCapture(0)

# Load known faces
laxman_image = face_recognition.load_image_file("C:/Users/omrat/OneDrive/Desktop/Projects/laxman.jpg")
laxman_encoding = face_recognition.face_encodings(laxman_image)[0]

om_image = face_recognition.load_image_file("C:/Users/omrat/OneDrive/Desktop/Projects/om.jpg")
om_encoding = face_recognition.face_encodings(om_image)[0]

known_face_encodings = [laxman_encoding, om_encoding]
known_face_names = ["Laxman", "Om"]

# List of expected students
students = known_face_names.copy()
face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open the csv file for writing
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

# Start the infinite loop to process frames from the video capture device
while True:
    # Read a frame from the video capture device
    _, frame = video_capture.read()

    # Resize the frame to 1/4 of its original size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize Faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Process each face detected in the frame
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Add the text if a person is present
            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

                # If the person is expected, remove from the list
                if name in students:
                    students.remove(name)
                    current_time = datetime.now().strftime("%H:%M:%S")  # Update to capture current time
                    lnwriter.writerow([name, current_time])

    # Display the processed frame with text annotations
    cv2.imshow("Camera", frame)

    # Check for the "q" key to exit the program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture device and destroy all windows
video_capture.release()
cv2.destroyAllWindows()

# Close the csv file
f.close()
