import cv2
import mediapipe as mp
import numpy as np

class PushUpCounter:
    def __init__(self):
        self.pushup_count = 0
        self.direction = 0  # 0: down, 1: up
        self.min_angle = 60
        self.max_angle = 150

        # Initialize Mediapipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def calculate_angle(self, a, b, c):
        a = np.array(a)  # First point
        b = np.array(b)  # Mid point
        c = np.array(c)  # End point

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def count_pushups(self, landmarks):
        # Get coordinates
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Calculate angle
        angle = self.calculate_angle(shoulder, elbow, wrist)

        # Push-up counting logic
        if angle > self.max_angle:
            if self.direction == 0:
                self.pushup_count += 1
                self.direction = 1
        if angle < self.min_angle:
            if self.direction == 1:
                self.direction = 0

        return angle

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor the frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detections
            results = self.pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                angle = self.count_pushups(landmarks)

                # Display push-up count
                cv2.putText(image, 'Push-ups: {}'.format(self.pushup_count),
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, 'Angle: {}'.format(round(angle, 2)),
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Render pose landmarks on the frame
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            except:
                pass

            # Display the frame
            cv2.imshow('Push-up Counter', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pushup_counter = PushUpCounter()
    pushup_counter.run()
