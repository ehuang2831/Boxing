import imageio
from PIL import Image
import numpy as np
import face_recognition

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Create list of frames where each frame is a 3 dimensional numpy array
filename = '/Users/Evan/Dropbox/Project/Pictures/Boxing Video/mayweather_mcgregor.mp4'
vid = imageio.get_reader(filename)
frame_shape = (np.asarray(vid.get_data(1))).shape
#frames_array = np.zeros(np.append(np.asarray(frame_shape),np.array(vid.get_length())))
frames = []
for num in range(9000,int(vid.get_length())):
    image = vid.get_data(num)
    frames.append(image)
    print('processed frame '+str(num)+'/'+str(vid.get_length()))
# frames = np.stack(frames,axis=-1)
# print('stacked frames into numpy array')

# initialize lists
face_locations = []
mayweather_loc = []
mcgregor_loc = []

# Load a sample picture of Mayweather and learn how to recognize it.
mayweather_image = face_recognition.load_image_file('/Users/Evan/Dropbox/Project/Pictures/Boxing Video/mayweather.jpg')
mayweather_face_encoding = face_recognition.face_encodings(mayweather_image)[0]

# Load a second sample picture of Mcgregor and learn how to recognize it.
mcgregor_image = face_recognition.load_image_file('/Users/Evan/Dropbox/Project/Pictures/Boxing Video/mcgregor.jpg')
mcgregor_face_encoding = face_recognition.face_encodings(mcgregor_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    mayweather_face_encoding,
    mcgregor_face_encoding
]
known_faces = [
    1, # Floyd Mayweather
    2 # Conor Mcgregor
]

# Grab the locations for every frame of Mayweather and Mcgregor
for frame in frames:
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        mcgregor_frame_loc = []
        mayweather_frame_loc = []
        if True in matches:
            first_match_index = matches.index(True)
            boxer_index = known_faces[first_match_index]
            if boxer_index == 1:
                mayweather_frame_loc = face_locations[first_match_index]
            elif boxer_index == 2:
                mcgregor_frame_loc = face_locations[first_match_index]


    mayweather_loc.append(mayweather_frame_loc)
    mcgregor_loc.append(mcgregor_frame_loc)
    print("processed frame " + str(frame) + "/" + str(len(frames)))