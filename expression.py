import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe FaceMesh and Drawing Utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Function to calculate Euclidean distance
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to analyze facial expressions
def analyze_expression(landmarks, width, height):
    # Extract key points for expression detection
    left_eye_inner = landmarks[133]  # Inner corner of left eye
    right_eye_inner = landmarks[362]  # Inner corner of right eye
    mouth_left = landmarks[61]  # Left corner of mouth
    mouth_right = landmarks[291]  # Right corner of mouth
    upper_lip = landmarks[13]  # Upper lip center
    lower_lip = landmarks[14]  # Lower lip center

    # Convert normalized coordinates to pixel coordinates
    def denormalize(point):
        return int(point.x * width), int(point.y * height)

    left_eye = denormalize(left_eye_inner)
    right_eye = denormalize(right_eye_inner)
    mouth_left = denormalize(mouth_left)
    mouth_right = denormalize(mouth_right)
    upper_lip = denormalize(upper_lip)
    lower_lip = denormalize(lower_lip)

    # Calculate distances
    eye_distance = calculate_distance(left_eye, right_eye)
    mouth_width = calculate_distance(mouth_left, mouth_right)
    mouth_height = calculate_distance(upper_lip, lower_lip)

    # Detect expressions based on distances
    if mouth_height / mouth_width > 0.5:
        return "Surprised"
    elif mouth_width / eye_distance > 1.8:
        return "Smiling"
    elif mouth_height / mouth_width < 0.15 and mouth_width / eye_distance < 1.5:
        return "Neutral"
    #elif mouth_height / mouth_width < 0.2 and eye_distance / mouth_width < 1.2:
        #return "Angry"
    else:
        return "Angry"

# Main function
def main():
    # Open webcam
    cap = cv2.VideoCapture(0)

    # Initialize FaceMesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            height, width, _ = frame.shape

            # Convert the frame to RGB as Mediapipe works with RGB images
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with FaceMesh
            results = face_mesh.process(rgb_frame)

            # Prepare a blank canvas for plotting points
            blank_canvas = np.zeros((height, width, 3), dtype=np.uint8)
            expression = "Neutral"

            # Draw face landmarks and analyze expression if detected
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                    # Extract landmark points
                    landmarks = face_landmarks.landmark

                    # Detect and display expression
                    expression = analyze_expression(landmarks, width, height)

                    # Highlight key points on the blank canvas
                    for lm in landmarks:
                        x, y = int(lm.x * width), int(lm.y * height)
                        cv2.circle(blank_canvas, (x, y), 2, (0, 255, 0), -1)

            # Combine both windows into one composite image
            composite = np.hstack((frame, blank_canvas))

            # Display expression on the combined window
            cv2.putText(composite, f"Expression: {expression}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show the composite image
            cv2.imshow('Face Mesh and Plotted Points', composite)

            # Exit on pressing 'Q' or 'ESC'
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key & 0xFF == 27:
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
