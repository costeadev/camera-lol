import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# CV2 is a library used to access to video capture devices and to show their output to a window
# Mediapipe is a library used to load AI tasks and use them to recognize images

# An AI model is like a "personality", it's a type of AI which is very good at something, in this case, hand detection
# "gesture_recognizer" is made by Google and it's trained with a large set of hand pictures.

# Load AI model task and create a recognizer (we will use this later to interpret video frames)
base_options = python.BaseOptions(model_asset_path="tasks/gesture_recognizer.task")
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Get the first available camera device
cap = cv2.VideoCapture(0)

# While the camera is being used
while cap.isOpened():
    
    # Read each frame of the camera
    ret, frame = cap.read() 
    
    # Ret is a boolean value which indicates if the frame is available
    # If it's not available we will skip it
    if not ret: break 
    
    # Flip the frame so the camera is not mirrored
    frame = cv2.flip(frame, 1)

    # Change the color of the frame from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use mediapipe to convert the frame into a valid image for the AI
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Use the recognizer to "recognize the image". 
    # This will create an object we can use to get information about each frame
    result = recognizer.recognize(mp_image)

    # We declare this variable to store to gesture being done in each frame
    gesture_name = "None"

    # Check if any gestures were found
    if result.gestures:
        # result.gestures is a list (one for each hand)
        for hand_gestures in result.gestures:
            # Get the top recognized gesture (The gesture which is most probable to be correct)
            top_gesture = hand_gestures[0]
            gesture_name = top_gesture.category_name # The category is a word like "Thumb_Up" which represents a gesture lol
            
            # Print it on the top of the window
            cv2.putText(frame, f"Gesture: {gesture_name}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # The recognizer converts each hand into 20 landmarks and stores them in a list
    # A landmark is a place in the hand, for instance, the wrist, which is landmark[0]
    if result.hand_landmarks: # (If the landamark exist)
        for hand in result.hand_landmarks:

            # Calculate the coordinates and draws the circles in the hand
            for lm in hand:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])

                # This draws the circle
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # This is a 2D array which stores pairs needed to trace a hand.
            # The first number is the start, the second is the end of the line
            bone_connections = [ 
                (0,1), (1,2), (2,3), (3,4),       # Thumb
                (0,5), (5,6), (6,7), (7,8),       # Index
                (5,9), (9,10), (10,11), (11,12),  # Middle
                (9,13), (13,14), (14,15), (15,16),# Ring
                (13,17), (17,18), (18,19), (19,20),# Pinky
                (0,17)                            # Palm base to pinky base
            ]

        # To draw a line, we need a starting point and an ending point
        # We will loop over our bone connections to calculate the position of each point on the screen
        for start_index, end_index in bone_connections:

            # Get coordinates for the start of the bone
            x1 = int(hand[start_index].x * frame.shape[1])
            y1 = int(hand[start_index].y * frame.shape[0])

            # Get coordinates for the end of the bone
            x2 = int(hand[end_index].x * frame.shape[1])
            y2 = int(hand[end_index].y * frame.shape[0])

            # This draws the line
            cv2.line(frame, (x1, y1), (x2, y2) , (255, 0, 0), 2)


    # Here we show each of the frames in the video in a separate window
    cv2.imshow("Gesture Control", frame)

    # We create another window for the meme based in the gesture
    cv2.namedWindow("meme")
    cv2.resizeWindow(winname="meme", width=500, height=500)
    cv2.moveWindow(winname="meme", x=960, y=320)

    # We make a map to map (duh) each gesture to its corresponding image
    gesture_pictures = {
        "Thumb_Up": "thumbsup.jpeg",
        "Pointing_Up": "nerd.jpg",
        "Open_Palm": "cinema.png",
        "Victory": "bye.jpeg",
        "Thumb_Down": "quetejodan.jpeg",
        "ILoveYou": "rock.jpeg",
        "Closed_Fist": "tepego.jpg",
        "None": "negro.png"
    }

    # This is the folder where out images are stored
    img_path = "assets/img/"

    # We access use key of "gesture_name" to access its value.
    # e.g. Thumb_Up --> thumbsup.jpeg
    picture_name = gesture_pictures[gesture_name]

    # We use the full path to the image to read it
    temp_img = cv2.imread(img_path + picture_name)

    # We resize temp_image and rename it to just 'img'
    img = cv2.resize(temp_img, (400, 400))

    # We check if the image exists, if it does, we show it inside the window 'meme'
    if img is not None:
        cv2.imshow("meme", img)
    else:
        print(f"Put {picture_name} in the !")

    # We make 'ESC' be the exit key for the window
    if cv2.waitKey(1) & 0XFF == 27:
        break
    
# Release the control of camera
cap.release()

# Removes all windows
cv2.destroyAllWindows()