import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculateAngle(landmark1, landmark2, landmark3):
    # Get the required landmarks coordinates.
    x1, y1 = landmark1
    x2, y2 = landmark2
    x3, y3 = landmark3

    # Calculate the angle between the three points
    angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
    angle = round(angle, 1)

    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360

    # Check if the angle greater than 180.
    if angle > 180.0:
        # subtract 360 from the found angle.
        angle = 360 - angle

    angle = 180 - angle

    # Return the calculated angle.
    return angle

def getCoords(landmarks, left_knee, right_knee):
    for (i, j) in zip(range(24, 30, 2), range(23, 29, 2)):
        right_knee.append([landmarks[mp_pose.PoseLandmark(i).value].x,
                                     landmarks[mp_pose.PoseLandmark(i).value].y])
        left_knee.append([landmarks[mp_pose.PoseLandmark(j).value].x,
                                    landmarks[mp_pose.PoseLandmark(j).value].y])
    return

def displayVisualization(knee_landmarks, connections, image, results, cap_size):
    # Calculate angle
    angle = calculateAngle(knee_landmarks[0], knee_landmarks[1], knee_landmarks[2])

    # Pose Detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, connections, landmark_drawing_spec=None)

    # Visualize angles
    image = cv2.putText(image, str(angle), tuple(np.multiply(knee_landmarks[1], cap_size).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return image, angle

def displayEndKeyInfo(image):
    image = cv2.putText(image, 'PRESS \'Q\' TO END MEASUREMENT', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def displayData(image, cap_size, min, max, angle_range):
    image = cv2.putText(image, f"MIN: {min}, MAX: {max}, DIFFERENCE: {angle_range}", (50, int(cap_size[1]) - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def validateData(angle, angles_vec):
    if not angles_vec:
        angles_vec.append(angle)
    else:
        prev_angle = angles_vec[len(angles_vec) - 1]
        if (abs(angle - prev_angle) < 20):
            angles_vec.append(angle)
    return


def CameraCapture(cap, angles_vec, leg_to_analyze, min, max, angle_range):
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:

        while cap.isOpened():
            _, frame = cap.read()

            # Change window name
            cv2.namedWindow('Knee angle measurment', cv2.WINDOW_NORMAL)

            # Get capture properties
            cap_properties = cv2.getWindowImageRect('Knee angle measurment')  # ( x, y, w, h)
            cap_size = (cap_properties[2], cap_properties[3])

            # Resize frame
            frame = cv2.resize(frame, cap_size, interpolation=cv2.INTER_AREA)

            # Recolor feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Make Detections
            results = pose.process(image)

            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Display text on frame
            image = displayEndKeyInfo(image)
            image = displayData(image, cap_size, min, max, angle_range)

            # Initialize variables
            right_knee_landmarks = []
            left_knee_landmarks = []
            leg_connections = ([(24, 26), (26, 28), (23, 25), (25, 27)])
            angle = 0


            # Check if any landmarks are detected.
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get landmarks coordinates
                getCoords(landmarks, left_knee_landmarks, right_knee_landmarks)

                # Selected leg measurment and visualization
                if (leg_to_analyze == 1):
                    leg_connections = (leg_connections[2], leg_connections[3])
                    image, angle = displayVisualization(left_knee_landmarks, leg_connections, image, results, cap_size)
                elif (leg_to_analyze == 2):
                    leg_connections = (leg_connections[0], leg_connections[1])
                    image, angle = displayVisualization(right_knee_landmarks, leg_connections, image, results, cap_size)

                # Data validation
                validateData(angle, angles_vec)
                min = np.min(angles_vec)
                max = np.max(angles_vec)
                angle_range = abs(max-min)

            cv2.imshow('Knee angle measurment', image)

            # Close camera capture if 'q' has been pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return min, max, angles_vec, angle_range

