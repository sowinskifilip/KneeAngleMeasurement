import math
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
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360

    # Return the calculated angle.
    return angle

# def drawLandmarks(frame, knee_landmarks):
#
#     line_color = (255, 255, 255)
#     line_width = 10
#
#     for el in knee_landmarks:
#         el[0] = int(el[0])
#         el[1] = int(el[1])
#
#     x1, y1 = knee_landmarks[0]
#     x2, y2 = knee_landmarks[1]
#     x3, y3 = knee_landmarks[2]
#
#     cv2.circle(frame, (x1, y1), 10, line_color, line_width)
#
#     return

def CameraCapture():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make Detections
            results = pose.process(image)
            # print(results.face_landmarks)

            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            right_knee_landmarks = []
            left_knee_landmarks = []
            leg_connections = ([(24, 26), (26, 28), (23, 25), (25, 27)])


            # Check if any landmarks are detected.
            if results.pose_landmarks:

                for (i, j) in zip(range(24,30,2), range(23,29,2)):
                    right_knee_landmarks.append([results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x,
                                               results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y])

                    left_knee_landmarks.append([results.pose_landmarks.landmark[mp_pose.PoseLandmark(j).value].x,
                                               results.pose_landmarks.landmark[mp_pose.PoseLandmark(j).value].y])

                right_angle = calculateAngle(right_knee_landmarks[0], right_knee_landmarks[1], right_knee_landmarks[2])
                left_angle = calculateAngle(left_knee_landmarks[0], left_knee_landmarks[1], left_knee_landmarks[2])
                # print('Left knee angle:', left_angle, '; Right knee angle:', right_angle)

                # Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, leg_connections, landmark_drawing_spec=None)

            cv2.imshow('Knee angle measurment', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return


def main():
    CameraCapture()

    return

if __name__ == '__main__':
    main()

