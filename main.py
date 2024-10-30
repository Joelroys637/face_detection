import face_recognition
import cv2
import numpy as np
import os
import streamlit as st
from datetime import date
import xlrd
from xlwt import Workbook
from xlutils.copy import copy as xl_copy
from PIL import Image

# Initialize Streamlit app
st.title("Live Face Recognition Attendance System")
st.write("Detect faces and mark attendance using webcam in real-time")

# Read current folder path for known images
CurrentFolder = os.getcwd()
image_path = os.path.join(CurrentFolder, 'leojoelroys.jpg')
image2_path = os.path.join(CurrentFolder, '24ucs554.jpg')

# Input subject lecture name
lecture_name = st.text_input('Enter subject lecture name:')
if not lecture_name:
    st.stop()

# Prepare Excel for attendance
file_path = 'attendance_excel.xls'
if not os.path.exists(file_path):
    wb = Workbook()
    sheet = wb.add_sheet("Sheet1")
    sheet.write(0, 0, 'Name/Date')
    sheet.write(0, 1, str(date.today()))
    wb.save(file_path)

rb = xlrd.open_workbook(file_path, formatting_info=True)
wb = xl_copy(rb)
sheet1 = wb.add_sheet(lecture_name)
sheet1.write(0, 0, 'Name/Date')
sheet1.write(0, 1, str(date.today()))
row = 1
col = 0
already_attendance_taken = ""

# Load known faces
person1_name = "leojoelroys"
person1_image = face_recognition.load_image_file(image_path)
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

person2_name = "Kenvin"
person2_image = face_recognition.load_image_file(image2_path)
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]

known_face_encodings = [person1_face_encoding, person2_face_encoding]
known_face_names = [person1_name, person2_name]

# Capture image from the webcam using Streamlit
captured_image = st.camera_input("Capture an image")

if captured_image is not None:
    # Convert the image from Streamlit to OpenCV format
    image = Image.open(captured_image)
    frame = np.array(image)

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Process the frame for face recognition
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        # Log attendance if recognized
        if (already_attendance_taken != name) and (name != "Unknown"):
            sheet1.write(row, col, name)
            col += 1
            sheet1.write(row, col, "Present")
            row += 1
            col = 0
            already_attendance_taken = name
            st.write(f"Attendance marked for {name}")
            wb.save(file_path)
        else:
            st.write("Next student")

    # Draw boxes around recognized faces
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the processed frame with annotations
    st.image(frame, channels="BGR")
import face_recognition
import cv2
import numpy as np
import os
import streamlit as st
from datetime import date
import xlrd
from xlwt import Workbook
from xlutils.copy import copy as xl_copy
from PIL import Image

# Initialize Streamlit app
st.title("Live Face Recognition Attendance System")
st.write("Detect faces and mark attendance using webcam in real-time")

# Read current folder path for known images
CurrentFolder = os.getcwd()
image_path = os.path.join(CurrentFolder, 'leojoelroys.jpg')
image2_path = os.path.join(CurrentFolder, '32bit.png')

# Input subject lecture name
lecture_name = st.text_input('Enter subject lecture name:')
if not lecture_name:
    st.stop()

# Prepare Excel for attendance
file_path = 'attendance_excel.xls'
if not os.path.exists(file_path):
    wb = Workbook()
    sheet = wb.add_sheet("Sheet1")
    sheet.write(0, 0, 'Name/Date')
    sheet.write(0, 1, str(date.today()))
    wb.save(file_path)

rb = xlrd.open_workbook(file_path, formatting_info=True)
wb = xl_copy(rb)
sheet1 = wb.add_sheet(lecture_name)
sheet1.write(0, 0, 'Name/Date')
sheet1.write(0, 1, str(date.today()))
row = 1
col = 0
already_attendance_taken = ""

# Load known faces
person1_name = "leojoelroys"
person1_image = face_recognition.load_image_file(image_path)
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

person2_name = "us-prime minister"
person2_image = face_recognition.load_image_file(image2_path)
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]

known_face_encodings = [person1_face_encoding, person2_face_encoding]
known_face_names = [person1_name, person2_name]

# Capture image from the webcam using Streamlit
captured_image = st.camera_input("Capture an image")

if captured_image is not None:
    # Convert the image from Streamlit to OpenCV format
    image = Image.open(captured_image)
    frame = np.array(image)

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Process the frame for face recognition
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        # Log attendance if recognized
        if (already_attendance_taken != name) and (name != "Unknown"):
            sheet1.write(row, col, name)
            col += 1
            sheet1.write(row, col, "Present")
            row += 1
            col = 0
            already_attendance_taken = name
            st.write(f"Attendance marked for {name}")
            wb.save(file_path)
        else:
            st.write("Next student")

    # Draw boxes around recognized faces
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the processed frame with annotations
    st.image(frame, channels="BGR")
