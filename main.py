import cv2
import mediapipe as mp
import pyautogui

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise Exception('Could not open video device')

#init facemesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks = True)

screen_w, screen_h = pyautogui.size()

#variable blink distance value
blink_threshold = 0.02
blink_counter = 0
blink_detected = False

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    frame_h, frame_w, _ = frame.shape #get frame dimensions

    if landmark_points :
        landmarks = landmark_points[0].landmark

        #Eye landmarks for mouse ctrl
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x,y), 3, (0,255,255))

            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

        #Blink detection for mouse click
        left_eye = [landmarks[145], landmarks[159]]
        for landmark in left_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x,y), 3, (0,255,255))

        #calculate eye-y distance
        y_difference = abs(left_eye[0].y -left_eye[1].y )
        cv2.putText(frame, f"Y Difference : {y_difference:4f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        #blink detection
        if y_difference < blink_threshold:
            blink_counter += 1
        else:
            blink_counter = 0

        if blink_counter > 5 and not blink_detected:
            pyautogui.click()
            blink_detected = True
            print('Blink detected! Mouse clicked')
        elif y_difference > blink_threshold:
            blink_detected = False

        if (left_eye[0].y - left_eye[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)  #delay to avoid multiple clicks

        cv2.imshow('Eye controlled mouse ', frame)

        # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()