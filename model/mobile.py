import random
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import base64
from flask_cors import CORS
from g_helper import bgr2rgb, mirrorImage
from fp_helper import pipelineHeadTiltPose, draw_face_landmarks_fp
from ms_helper import pipelineMouthState
from es_helper import pipelineEyesState
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
from deepface import DeepFace
from pymongo import MongoClient
from twilio.rest import Client

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Twilio setup

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# MongoDB setup
client = MongoClient("mongodb+srv://vibudesh:040705@cluster0.bojv6ut.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["SIH"]
collection = db["face"]

# Face detection and anti-spoofing models
face_detector = YOLOv5('saved_models/yolov5s-quantized.onnx')
anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')

mp_face_mesh = mp.solutions.face_mesh

instructions = [
    "Turn your head left",
    "Turn your head right",
    "Look up",
    "Look down",
    "Open your mouth"
]

current_instruction = random.choice(instructions)
action_counts = {
    "left": 0,
    "right": 0,
    "up": 0,
    "down": 0,
    "mouthOpen": 0
}

client_aadhar_map = {}

def send_spoofing_alert(phone_number):
    try:
        message = twilio_client.messages.create(
            body="ALERT: Your account may have been spoofed. Please contact support immediately.",
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        print(f"[DEBUG] Spoofing alert sent to {phone_number}: {message.sid}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send spoofing alert to {phone_number}: {str(e)}")
        return False

def fetch_known_face(aadhaar_number):
    user_data = collection.find_one({"roll_number": aadhaar_number})
    if not user_data:
        raise ValueError(f"Aadhaar number {aadhaar_number} not found in the database.")
    
    binary_data = user_data["image"]
    base64_data = base64.b64encode(binary_data).decode('utf-8')
    img_data = base64.b64decode(base64_data)
    np_arr = np.frombuffer(img_data, np.uint8)
    known_face_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return known_face_img

def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    real_h, real_w = img.shape[:2]
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    xc, yc = x + w / 2, y + h / 2
    x, y = int(xc - l * bbox_inc / 2), int(yc - l * bbox_inc / 2)
    x1 = 0 if x < 0 else x
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l * bbox_inc > real_w else x + int(l * bbox_inc)
    y2 = real_h if y + l * bbox_inc > real_h else y + int(l * bbox_inc)
    img = img[y1:y2, x1:x2, :]
    img = cv2.copyMakeBorder(img, y1 - y, int(l * bbox_inc - y2 + y), x1 - x, int(l * bbox_inc) - x2 + x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def make_prediction(img, face_detector, anti_spoof):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox = face_detector([img])[0]
    if bbox.shape[0] > 0:
        bbox = bbox.flatten()[:4].astype(int)
    else:
        return None
    pred = anti_spoof([increased_crop(img, bbox, bbox_inc=1.5)])[0]
    score = pred[0][0]
    label = np.argmax(pred)
    return bbox, label, score

def decode_base64_image(image_base64):
    image_data = base64.b64decode(image_base64.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def process_image(image, aadhaar_number):
    client_id = request.sid
    print(f"[DEBUG] Client ID: {client_id}")

    client_data = client_aadhar_map.get(client_id)
    if not client_data or "aadhaar_number" not in client_data:
        print(f"[DEBUG] Aadhaar number not found in session for client {client_id}")
        return "Aadhaar number not found in session."

    aadhaar_number = client_data["aadhaar_number"]
    
    try:
        print(f"[DEBUG] Fetching known face for Aadhaar number: {aadhaar_number}")
        known_face_img = fetch_known_face(aadhaar_number)  
    except ValueError as e:
        print(f"[ERROR] {str(e)}")
        return str(e) 
    
    print("[DEBUG] Starting face prediction...")
    pred = make_prediction(image, face_detector, anti_spoof)
    if pred is not None:
        (x1, y1, x2, y2), label, score = pred
        print(f"[DEBUG] Predicted face coordinates: ({x1}, {y1}), ({x2}, {y2})")
        print(f"[DEBUG] Prediction label: {label}, score: {score}")
        
        face_crop = image[y1:y2, x1:x2]
      
        if label == 0 and score > 0.5:  # Ensuring it's a real face
            try:
                print("[DEBUG] Verifying face with DeepFace...")
                result = DeepFace.verify(face_crop, known_face_img, model_name="VGG-Face")
                print(f"[DEBUG] DeepFace verification result: {result}")
                
                if result["verified"]:
                    print("[DEBUG] Face is REAL and MATCHED")
                    return "REAL and MATCHED"
                else:
                    print("[DEBUG] Face does not match")
                    return "NOT MATCHING"
            except Exception as e:
                print(f"[ERROR] Error during DeepFace verification: {str(e)}")
                return f"Processing error: {str(e)}"
        else:
            print("[DEBUG] Face is FAKE")
            return "FAKE"
    else:
        print("[DEBUG] No face detected")
    return "No Face Detected"

@app.route('/')
def index():
    return "Face Pose Tracking Server is running!"

@socketio.on('send_aadhaar')
def handle_aadhaar(data):
    aadhaar_number = data.get('aadhaar')
    print(f"Received Aadhaar Number: {aadhaar_number}")
    client_aadhar_map[request.sid] = {"aadhaar_number": aadhaar_number}

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    emit("receive_instruction", {"instruction": current_instruction, "action_counts": action_counts})

@socketio.on("send_frame")
def process_frame(data):
    global current_instruction, action_counts

    print("[DEBUG] Processing frame...")

    try:
        frame_data = base64.b64decode(data.split(",")[1])
        np_image = np.frombuffer(frame_data, dtype=np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] Decoding frame failed: {e}")
        return

    image = mirrorImage(image)

    face_match_result = process_image(image, client_aadhar_map[request.sid]['aadhaar_number'])
    print(f"[DEBUG] Face match result: {face_match_result}")
    emit('validate_face_matching', {"status": face_match_result})

    if face_match_result == "FAKE":
        aadhaar_number = client_aadhar_map[request.sid]['aadhaar_number']
        user_data = collection.find_one({"roll_number": aadhaar_number})
        if user_data and 'phone' in user_data:
            phone_number = user_data['phone']
            emit('fake_face_detected', {"aadhaar_number": aadhaar_number, "phone_number": phone_number})
        else:
            print(f"[ERROR] Phone number not found for Aadhaar: {aadhaar_number}")
            emit('fake_face_detected', {"aadhaar_number": aadhaar_number, "error": "Phone number not found"})


    pred = make_prediction(image, face_detector, anti_spoof)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:
        try:
            results = face_mesh.process(bgr2rgb(image))
        except Exception as e:
            print(f"[ERROR] Mediapipe processing failed: {e}")
            return

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                draw_face_landmarks_fp(image, face_landmarks)
                head_tilt_pose = pipelineHeadTiltPose(image, face_landmarks)
                mouth_state = pipelineMouthState(image, face_landmarks)
                r_eyes_state, l_eyes_state = pipelineEyesState(image, face_landmarks)

                print(f"[DEBUG] Current Instruction: {current_instruction}")
                print(f"[DEBUG] Detected Pose: {head_tilt_pose}, Mouth State: {mouth_state}")

                correct_action = False
                if current_instruction == "Turn your head left" and head_tilt_pose == "Left":
                    correct_action = True
                elif current_instruction == "Turn your head right" and head_tilt_pose == "Right":
                    correct_action = True
                elif current_instruction == "Look up" and head_tilt_pose == "Up":
                    correct_action = True
                elif current_instruction == "Look down" and head_tilt_pose == "Down":
                    correct_action = True
                elif current_instruction == "Open your mouth" and mouth_state == "Open":
                    correct_action = True

                if correct_action:
                    if current_instruction == "Turn your head left":
                        action_counts["left"] += 1
                    elif current_instruction == "Turn your head right":
                        action_counts["right"] += 1
                    elif current_instruction == "Look up":
                        action_counts["up"] += 1
                    elif current_instruction == "Look down":
                        action_counts["down"] += 1
                    elif current_instruction == "Open your mouth":
                        action_counts["mouthOpen"] += 1

                    print(f"[DEBUG] Updated Action Counts: {action_counts}")

                    total_count = sum(action_counts.values())

                    if total_count > 5:
                        print("[DEBUG] Total action count greater than 5. All actions completed.")
                        emit("actions_completed", {"status": "success", "message": "All actions completed"})
                        return

                    next_instruction = random.choice(instructions)
                    while next_instruction == current_instruction:
                        next_instruction = random.choice(instructions)

                    current_instruction = next_instruction
                    print(f"[DEBUG] Next Instruction: {current_instruction}")

                    emit("receive_instruction", {"instruction": current_instruction, "action_counts": action_counts})

@app.route('/fetch-user-data', methods=['POST'])
def fetch_user_data():
    try:
        data = request.json
        
        # Log incoming request data for debugging
        app.logger.debug(f"Received request data: {data}")
        
        aadhaar_number = data.get("aadhaar")
        if not aadhaar_number:
            app.logger.error("Aadhaar number not provided.")
            return {"success": False, "message": "Aadhaar number not provided."}, 400

        # Query the database for user data
        user_data = collection.find_one({"roll_number": aadhaar_number})

        if not user_data:
            app.logger.error(f"User not found for Aadhaar: {aadhaar_number}")
            return {"success": False, "message": "User not found."}, 404

        name = user_data.get("name")
        binary_image = user_data.get("aadhar")  # Assuming "aadhar" field holds the image

        # Check if the necessary fields exist in the response
        if not name or not binary_image:
            app.logger.error(f"Incomplete data for Aadhaar: {aadhaar_number}, name: {name}, image: {binary_image}")
            return {"success": False, "message": "Incomplete user data."}, 400

        # If the image is stored as a base64 string, decode it to bytes
        if isinstance(binary_image, str):
            try:
                binary_image = base64.b64decode(binary_image)
            except Exception as e:
                app.logger.error(f"Error decoding base64 image for Aadhaar {aadhaar_number}: {str(e)}")
                return {"success": False, "message": "Error decoding base64 image."}, 400

        # Encode the image to base64
        image_base64 = base64.b64encode(binary_image).decode('utf-8')

        app.logger.debug(f"Successfully fetched data for Aadhaar: {aadhaar_number}")

        # Return the response
        return {"success": True, "name": name, "image": image_base64}, 200

    except Exception as e:
        # Log the exception details for debugging
        app.logger.error(f"An error occurred: {str(e)}")
        return {"success": False, "message": str(e)}, 500

@app.route('/send-sms', methods=['POST'])
def send_sms():
    data = request.json
    aadhaar_number = data.get('aadhaar')
    print(aadhaar_number)
    if not aadhaar_number:
        return jsonify({"success": False, "message": "Aadhaar number not provided"}), 400
    
    user_data = collection.find_one({"roll_number": aadhaar_number})
    if not user_data or 'phone' not in user_data:
        return jsonify({"success": False, "message": "Phone number not found for the given Aadhaar number"}), 404
    
    phone_number = user_data['phone']
    print(phone_number)
    if send_spoofing_alert(phone_number):
        return jsonify({"success": True, "message": "Spoofing alert sent successfully"}), 200
    else:
        return jsonify({"success": False, "message": "Failed to send spoofing alert"}), 500

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)

