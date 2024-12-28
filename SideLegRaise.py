import os
import cv2
import mediapipe as mp
import numpy as np
import absl.logging

# Suppress verbose logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
absl.logging.set_verbosity(absl.logging.ERROR)

# Preload TensorFlow and MediaPipe models
def preload_models():
    print("Preloading models...")
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        dummy_image = np.zeros((128, 128, 3), dtype=np.uint8)  # Warm up
        pose.process(dummy_image)
    print("Models preloaded.")

# Preload models at the start
preload_models()

# Load MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def get_leg_keypoints(landmarks, side="left"):
    """Extract keypoints for the specified leg side ('left' or 'right')."""
    if side == "left":
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    else:
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
    return shoulder, hip, knee, ankle

def process_leg(hip, knee, ankle, shoulder, stage, counter):
    """Process leg rep counting and feedback based on joint angles."""
    leg_raise_angle = calculate_angle(shoulder, hip, knee)
    knee_straightness_angle = calculate_angle(hip, knee, ankle)

    ideal_leg_raise_range = (90, 140)
    ideal_knee_straightness = 170
    
    if (ideal_leg_raise_range[0] <= leg_raise_angle <= ideal_leg_raise_range[1]
        and knee_straightness_angle > ideal_knee_straightness and stage == "down"):
        stage = "up"
        counter += 1
    elif leg_raise_angle > 170 and stage == "up" and knee_straightness_angle > ideal_knee_straightness:
        stage = "down"

    if leg_raise_angle > 140:
        feedback = "Raise your leg higher to the side."
    elif leg_raise_angle < 90:
        feedback = "Don't raise leg too high!"
    elif knee_straightness_angle < 170:
        feedback = "Keep your leg straight."
    else:
        feedback = "Good form!"

    return counter, stage, feedback

def display_info(image, left_counter, left_stage, right_counter, right_stage, left_feedback, right_feedback):
    """Display counters, stages, and feedback for both legs."""
    # Display counter and stage for left leg at the top left
    cv2.rectangle(image, (10, 10), (170, 50), (245, 117, 16), -1)
    cv2.putText(image, f"LEFT REPS: {left_counter}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"LEFT STAGE: {left_stage}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Feedback for left leg with dynamic background width
    (text_w, text_h), _ = cv2.getTextSize(left_feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (10, 60), (20 + text_w, 90), (0, 0, 0), -1)
    cv2.putText(image, left_feedback, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Display counter and stage for right leg at bottom left
    height = image.shape[0]
    cv2.rectangle(image, (10, height - 90), (170, height - 50), (245, 117, 16), -1)
    cv2.putText(image, f"RIGHT REPS: {right_counter}", (15, height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"RIGHT STAGE: {right_stage}", (15, height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Feedback for right leg with dynamic background width, adjusted vertically to avoid overlap
    (text_w, text_h), _ = cv2.getTextSize(right_feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (10, height - 130), (20 + text_w, height - 100), (0, 0, 0), -1)
    cv2.putText(image, right_feedback, (15, height - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

cap = cv2.VideoCapture(1)
left_counter, right_counter = 0, 0
left_stage, right_stage = "down", "down"
left_feedback, right_feedback = "", ""

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Create a named window and set it to always be on top
cv2.namedWindow("Side Leg Raise Counter", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Side Leg Raise Counter", cv2.WND_PROP_TOPMOST, 1)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Warning: Unable to read a frame from the camera.")
            break

        # Recolor image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Process left leg
            left_shoulder, left_hip, left_knee, left_ankle = get_leg_keypoints(landmarks, side="left")
            left_counter, left_stage, left_feedback = process_leg(left_hip, left_knee, left_ankle, left_shoulder, left_stage, left_counter)

            # Process right leg
            right_shoulder, right_hip, right_knee, right_ankle = get_leg_keypoints(landmarks, side="right")
            right_counter, right_stage, right_feedback = process_leg(right_hip, right_knee, right_ankle, right_shoulder, right_stage, right_counter)

        # Display info for both legs
        display_info(image, left_counter, left_stage, right_counter, right_stage, left_feedback, right_feedback)

        # Render pose landmarks on the frame
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow("Side Leg Raise Counter", image)

        # Exit conditions
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # Check if the window was closed
        if cv2.getWindowProperty("Side Leg Raise Counter", cv2.WND_PROP_VISIBLE) < 1:
            break

    # Cleanup resources after loop
    cap.release()
    cv2.destroyAllWindows()
