import cv2 as cv
import dlib
from PIL import Image
from torchvision import transforms
import threading
import torch
import time
from cnn_model import DrowsinessCNN
import torch.nn as nn
import numpy as np

# Capturing the video input from webcam.
cap = cv.VideoCapture(0)

# Global Variables
right_eye = None
left_eye = None

# Model output variables.
output_right = None
output_left = None

detector = dlib.get_frontal_face_detector()
# IMPORTANT
# Inside of the shape_predictor function , we must define the predictor
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

# Input format of the model.
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1))
])

# To load the parameters of the trained model, first we need to initialize same model that we have created into the running application.
model = DrowsinessCNN()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_string = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Loading the parameters of the model - WEIGHTS, BIASES and more.
model.load_state_dict(torch.load('./saved_model/drowsiness.pth'))
model.eval()

# Input data for the prediction
def define_input(right_eye_input, left_eye_input):

    global right_eye
    global left_eye

    right_eye_image = Image.fromarray(right_eye_input)
    right_eye = right_eye_image.resize((145, 145))

    left_eye_image = Image.fromarray(left_eye_input)
    left_eye = left_eye_image.resize((145, 145))

    right_eye = input_transform(right_eye)
    left_eye = input_transform(left_eye)

# Output data of the prediction
def model_output(right_eye, left_eye):

    global output_right
    global output_left

    with torch.no_grad():
        output_right = model(right_eye)
        output_left = model(left_eye)
        m = nn.Sigmoid()

        output_right = m(output_right)
        output_left = m(output_left)

        output_right = output_right.numpy()
        output_left = output_left.numpy()

while True:

    _, frame = cap.read()

    # Get data from VideoCapture(0) - must be in gray format.
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect the face inside of the frame
    faces = detector(gray)

    # Iterate faces.
    for face in faces:

        # Apply the landmark to the detected face
        face_lndmrks = predictor(gray, face)

        righteye_x_strt = face_lndmrks.part(37).x-25
        righteye_x_end = face_lndmrks.part(40).x+25
        righteye_y_strt = face_lndmrks.part(20).y-10
        righteye_y_end = face_lndmrks.part(42).y+20

        lefteye_x_strt = face_lndmrks.part(43).x-25
        lefteye_x_end = face_lndmrks.part(46).x+25
        lefteye_y_strt = face_lndmrks.part(25).y-10
        lefteye_y_end = face_lndmrks.part(47).y+20
        cv.rectangle(frame, (righteye_x_strt, righteye_y_strt),
                     (righteye_x_end, righteye_y_end), (0, 255, 0), 2)
        cv.rectangle(frame, (lefteye_x_strt, lefteye_y_strt),
                     (lefteye_x_end, lefteye_y_end), (0, 255, 0), 2)

        right_eye_input = frame[righteye_y_strt:righteye_y_end,
                                righteye_x_strt:righteye_x_end]
        left_eye_input = frame[lefteye_y_strt:lefteye_y_end,
                               lefteye_x_strt:lefteye_x_end]
        
        current_time = time.localtime().tm_sec
        
        a = threading.Thread(target = define_input, args=(
            right_eye_input, left_eye_input))
        b = threading.Thread(target = model_output, args = (right_eye, left_eye))
    
        a.start()
        b.start()
    
        a.join()
        b.join()
            
        # Prediction of the eyes whether closed or opened
        drowsiness = []
        if output_left != None:
            drowsiness.append(output_left)
            drowsiness = np.asarray(drowsiness)
            
        if current_time % 10 == 0:
            try:
                mean_drows = sum(drowsiness) / len(drowsiness)
                
                if mean_drows < 0.5:
                     predict = "Awake"
                else:
                     predict = "Sleepy"
                cv.putText(frame, predict, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

            except ZeroDivisionError:
                pass

    # Display in the frame
    cv.imshow('Frame', frame)

    if cv.waitKey(1) == ord("q"):
        cap.release()
        cv.destroyAllWindows()
        break