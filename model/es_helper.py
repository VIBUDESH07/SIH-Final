import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def getCoordinates_es(idx, landmark, img_h, img_w, r_upper_loc, r_upper_x, r_lower_loc, r_lower_x, l_upper_loc, l_upper_x, l_lower_loc, l_lower_x):
    x, y = landmark.x, landmark.y
    x_on_image = int(x*img_w)
    y_on_image = int(y*img_h)
    
    r_upper_eyes_index = 159
    r_lower_eyes_index = 145
    l_upper_eyes_index = 386
    l_lower_eyes_index = 374

    if idx == r_upper_eyes_index:
        r_upper_loc = y_on_image
        r_upper_x = x_on_image
    elif idx == r_lower_eyes_index:
        r_lower_loc = y_on_image
        r_lower_x = x_on_image
    elif idx == l_upper_eyes_index:
        l_upper_loc = y_on_image
        l_upper_x = x_on_image
    elif idx == l_lower_eyes_index:
        l_lower_loc = y_on_image
        l_lower_x = x_on_image
        
    return r_upper_loc, r_upper_x, r_lower_loc, r_lower_x, l_upper_loc, l_upper_x, l_lower_loc, l_lower_x

def getEyesState_es(r_upper_loc, r_lower_loc, l_upper_loc, l_lower_loc):
    r_distance = int(r_lower_loc - r_upper_loc)
    l_distance = int(l_lower_loc - l_upper_loc)
    
    if r_distance > 6:
        r_eyesStatus = 'Left Eye Opened'
    else:
        r_eyesStatus = 'Left Eye Closed'
    if l_distance > 6:
        l_eyesStatus = 'Right Eye Opened'
    else:
        l_eyesStatus = 'Right Eye Closed'
    return r_eyesStatus, r_distance, l_eyesStatus, l_distance

def draw_eyes_eyes_dots_es(image, x, y):
    cv2.circle(img=image, center=(x, y), radius = 3, color=(0,255,0))

def draw_eyes_state_es(image, r_text, l_text):
    cv2.putText(image, f"EYES: {r_text}) | {l_text})", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

def draw_eyes_condition_es(image, r_upper_eye_loc, r_lower_eye_loc, r_distance, l_upper_eye_loc, l_lower_eye_loc, l_distance):
    cv2.putText(image, f"(R) upper/lower: {r_upper_eye_loc}/{r_lower_eye_loc} - distance: {r_distance}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(image, f"(L) upper/lower: {l_upper_eye_loc}/{l_lower_eye_loc} - distance: {l_distance}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def pipelineEyesState(image, face_landmarks):
    
    img_h, img_w, img_c = image.shape
    landmark_points = face_landmarks.landmark
    
    r_upper_loc = 0
    r_upper_x = 0
    r_lower_loc = 0
    r_lower_x = 0
    
    l_upper_loc = 0
    l_upper_x = 0
    l_lower_loc = 0
    l_lower_x = 0
    for idx, landmark in enumerate(landmark_points):
        
        r_upper_loc, r_upper_x, r_lower_loc, r_lower_x, l_upper_loc, l_upper_x, l_lower_loc, l_lower_x = getCoordinates_es(
            idx, landmark, img_h, img_w, 
            r_upper_loc, r_upper_x, 
            r_lower_loc, r_lower_x,
            l_upper_loc, l_upper_x, 
            l_lower_loc, l_lower_x)
    
    r_eyes_state, r_distance, l_eyes_state, l_distance = getEyesState_es(r_upper_loc, r_lower_loc, l_upper_loc, l_lower_loc)
    
    draw_eyes_state_es(image, r_eyes_state, l_eyes_state)
    draw_eyes_condition_es(image, r_upper_loc, r_lower_loc, r_distance, l_upper_loc, l_lower_loc, l_distance)
    draw_eyes_eyes_dots_es(image, r_upper_x, r_upper_loc)
    draw_eyes_eyes_dots_es(image, r_lower_x, r_lower_loc)
    draw_eyes_eyes_dots_es(image, l_upper_x, l_upper_loc)
    draw_eyes_eyes_dots_es(image, l_lower_x, l_lower_loc)

    return r_eyes_state, l_eyes_state