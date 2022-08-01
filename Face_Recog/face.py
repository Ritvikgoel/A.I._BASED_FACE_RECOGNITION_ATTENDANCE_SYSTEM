import face_recognition
import cv2
import numpy as np
import pandas as pd
import datetime


video_capture = cv2.VideoCapture(0)

def MarkAttendence(name):
    sheet = pd.read_csv('attendence.csv')

    pos =0
    for i in range(0,len(sheet.index)):
        if sheet.UID[i]==name:
            pos = i
    if sheet.Present[pos] != 1 and sheet.UID[pos]==name:
        date = datetime.datetime.now()
        date= date.strftime("%x")
        sheet.Date[pos] = date
        sheet.Present[pos] = 1
        sheet.to_csv('attendence.csv',index=False)
        

my_image = face_recognition.load_image_file("dataset/3510.jpg")
my_face_encoding = face_recognition.face_encodings(my_image)[0]


pep1_image = face_recognition.load_image_file("dataset/3511.jpg")
pep1_face_encoding = face_recognition.face_encodings(pep1_image)[0]

pep2_image = face_recognition.load_image_file("dataset/3512.jpg")
pep2_face_encoding = face_recognition.face_encodings(pep2_image)[0]

pep3_image = face_recognition.load_image_file("dataset/3531.jpg")
pep3_face_encoding = face_recognition.face_encodings(pep3_image)[0]

pep4_image = face_recognition.load_image_file("dataset/3514.jpeg")
pep4_face_encoding = face_recognition.face_encodings(pep4_image)[0]

pep5_image = face_recognition.load_image_file("dataset/3515.jpg")
pep5_face_encoding = face_recognition.face_encodings(pep5_image)[0]

pep6_image = face_recognition.load_image_file("dataset/3516.jpg")
pep6_face_encoding = face_recognition.face_encodings(pep6_image)[0]

pep7_image = face_recognition.load_image_file("dataset/3517.jpg")
pep7_face_encoding = face_recognition.face_encodings(pep7_image)[0]

pep8_image = face_recognition.load_image_file("dataset/3518.jpg")
pep8_face_encoding = face_recognition.face_encodings(pep8_image)[0]

pep9_image = face_recognition.load_image_file("dataset/3519.jpg")
pep9_face_encoding = face_recognition.face_encodings(pep9_image)[0]

pep10_image = face_recognition.load_image_file("dataset/3520.jpg")
pep10_face_encoding = face_recognition.face_encodings(pep10_image)[0]

pep11_image = face_recognition.load_image_file("dataset/3521.jpg")
pep11_face_encoding = face_recognition.face_encodings(pep11_image)[0]

pep12_image = face_recognition.load_image_file("dataset/3522.jpg")
pep12_face_encoding = face_recognition.face_encodings(pep12_image)[0]

pep13_image = face_recognition.load_image_file("dataset/3523.jpg")
pep13_face_encoding = face_recognition.face_encodings(pep13_image)[0]

pep14_image = face_recognition.load_image_file("dataset/3524.jpg")
pep14_face_encoding = face_recognition.face_encodings(pep14_image)[0]

pep15_image = face_recognition.load_image_file("dataset/3545.jpeg")
pep15_face_encoding = face_recognition.face_encodings(pep15_image)[0]

pep16_image = face_recognition.load_image_file("dataset/Navin.jpeg")
pep16_face_encoding = face_recognition.face_encodings(pep16_image)[0]

known_face_encodings = [
    my_face_encoding, pep1_face_encoding, pep2_face_encoding, pep3_face_encoding, pep4_face_encoding, pep5_face_encoding,
    pep6_face_encoding, pep7_face_encoding, pep8_face_encoding, pep9_face_encoding, pep10_face_encoding, pep11_face_encoding
, pep12_face_encoding, pep13_face_encoding, pep14_face_encoding, pep15_face_encoding,pep16_face_encoding
]

known_face_names = [
    "19BCS3510","19BCS3511","19BCS3512","19BCS3531","19BCS3514","19BCS3515","19BCS3516","19BCS3517","19BCS3518","19BCS3519"
,"19BCS3520","19BCS3521","19BCS3522","19BCS3523","19BCS3524","19BCS3545","Navin"
]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
  
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
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
            MarkAttendence(name)

    process_this_frame = not process_this_frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
