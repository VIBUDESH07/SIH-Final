import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def getCoordinates_ms(idx, landmark, img_h, img_w, upper_loc, upper_x, lower_loc, lower_x):
    x, y = landmark.x, landmark.y
    x_on_image = int(x*img_w)
    y_on_image = int(y*img_h)
    
    upper_lip_index = 0
    lower_lip_index = 14
    if idx == upper_lip_index:
        upper_loc = y_on_image
        upper_x = x_on_image
    elif idx == lower_lip_index:
        lower_loc = y_on_image
        lower_x = x_on_image
    return upper_loc, upper_x, lower_loc, lower_x

def getMouthState_ms(upper_lip_loc, lower_lip_loc):
    distance = int(lower_lip_loc - upper_lip_loc)
    if distance > 20:
        mouthStatus = 'Open'
    else:
        mouthStatus = 'Close'
    return mouthStatus, distance

def draw_mouth_lips_dots_ms(image, x, y):
    cv2.circle(img=image, center=(x, y), radius = 3, color=(0,255,0))

def draw_mouth_state_ms(image, text):
    cv2.putText(image, f"MOUTH: {text}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

def draw_mouth_condition_ms(image, upper_lip_loc, lower_lip_loc, distance):
    cv2.putText(image, f"upper/lower: {upper_lip_loc}/{lower_lip_loc}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(image, f"distance: {distance}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

def pipelineMouthState(image, face_landmarks):
    
    img_h, img_w, img_c = image.shape
    landmark_points = face_landmarks.landmark
    upper_lip_loc = 0
    upper_x = 0
    lower_lip_loc = 0
    lower_x = 0
    for idx, landmark in enumerate(landmark_points):
        
        upper_lip_loc, upper_x, lower_lip_loc, lower_x = getCoordinates_ms(idx, landmark, img_h, img_w, upper_lip_loc, upper_x, lower_lip_loc, lower_x)
    
    mouthState, distance = getMouthState_ms(upper_lip_loc, lower_lip_loc)
    
    draw_mouth_state_ms(image, mouthState)
    draw_mouth_condition_ms(image, upper_lip_loc, lower_lip_loc, distance)
    draw_mouth_lips_dots_ms(image, upper_x, upper_lip_loc)
    draw_mouth_lips_dots_ms(image, lower_x, lower_lip_loc)
    return mouthState