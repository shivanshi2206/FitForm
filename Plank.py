import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def is_person_lying_down(landmarks):
    shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
    hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x
    ankle_x = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x

    if np.abs(shoulder_x - hip_x) < 0.1 and np.abs(hip_x - ankle_x) < 0.1:
        return "Lying down"
    else:
        return "Standing"


cap = cv2.VideoCapture(1)
start_time = None
timer_running = False
elapsed_time = 0
# Create a named window and set it to always be on top
cv2.namedWindow("Plank Counter", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Plank Counter", cv2.WND_PROP_TOPMOST, 1)
# Optional: Set the window to full screen
#cv2.setWindowProperty("Plank Counter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Set the window name

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor image to RGB for processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate key angles
            shoulder_angle = calculate_angle(shoulder, elbow, wrist)
            hip_angle = calculate_angle(shoulder, hip, knee)
            knee_angle = calculate_angle(hip, knee, ankle)
            hip_shoulder_elbow_angle = calculate_angle(hip, shoulder, elbow)

            # Define ideal angle ranges
            ideal_shoulder_angle_range = (60, 90)  # Shoulder alignment
            ideal_hip_angle_range = (150, 180)     # Hip alignment for flat back
            ideal_knee_angle_range = (150, 180)    # Knee alignment for leg extension
            ideal_hip_shoulder_elbow_range = (60, 100)  # New angle range

            # Initialize correction message
            correction_message = ""

            if is_person_lying_down(landmarks)=="Standing":
                timer_running = False
                elapsed_time = 0
                correction_message = "Lay down."

            # Combine checks for proper plank position
            if (ideal_shoulder_angle_range[0] <= shoulder_angle <= ideal_shoulder_angle_range[1] and
                ideal_hip_angle_range[0] <= hip_angle <= ideal_hip_angle_range[1] and
                ideal_knee_angle_range[0] <= knee_angle <= ideal_knee_angle_range[1] and
                ideal_hip_shoulder_elbow_range[0] <= hip_shoulder_elbow_angle <= ideal_hip_shoulder_elbow_range[1]):

                if not timer_running:
                    start_time = time.time()
                    timer_running = True
                elapsed_time = int(time.time() - start_time)
                correction_message = "Good posture!"
            else:
                # Reset timer if posture is incorrect
                timer_running = False
                elapsed_time = 0

                # Generate specific correction messages based on angle deviations
                if not (ideal_shoulder_angle_range[0] <= shoulder_angle <= ideal_shoulder_angle_range[1]):
                    correction_message = "Adjust your shoulder alignment."
                elif not (ideal_hip_angle_range[0] <= hip_angle <= ideal_hip_angle_range[1]):
                    correction_message = "Keep your back straight."
                elif not (ideal_knee_angle_range[0] <= knee_angle <= ideal_knee_angle_range[1]):
                    correction_message = "Straighten your legs."
                elif not (ideal_hip_shoulder_elbow_range[0] <= hip_shoulder_elbow_angle <= ideal_hip_shoulder_elbow_range):
                    correction_message = "Adjust your hip-shoulder-elbow alignment."

            # Display timer and correction messages
            cv2.rectangle(image, (0, 0), (300, 90), (245, 117, 16), -1)
            cv2.putText(image, "TIMER", (15, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f"{elapsed_time} sec", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display correction message
            cv2.putText(image, correction_message, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255), 2, cv2.LINE_AA)
            

            # Display calculated angles
            cv2.putText(image, str(int(shoulder_angle)), tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (2, 186, 30), 2, cv2.LINE_AA)
            cv2.putText(image, str(int(hip_angle)), tuple(np.multiply(hip, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (2, 186, 30), 2, cv2.LINE_AA)
            cv2.putText(image, str(int(knee_angle)), tuple(np.multiply(knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (2, 186, 30), 2, cv2.LINE_AA)
            cv2.putText(image, str(int(hip_shoulder_elbow_angle)), tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (2, 186, 30), 2, cv2.LINE_AA)

        except:
            correction_message = "No user detected"
            timer_running = False
            elapsed_time = 0
            cv2.putText(image, correction_message, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # Render pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the image in fullscreen
        cv2.imshow("Plank Counter", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Plank Counter", cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()