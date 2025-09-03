import cv2
import numpy as np
import random, time, os
from HandTrackingModule import handDetector  # import your module

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = handDetector(detectionCon=0.8)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Word prompts
prompts = ["cat", "house", "tree", "robot", "car", "pizza", "dog", "fish",
            "ball", "lamp", "desk", "cloud", "tree", "plant", "moon",
            "star", "door", "face", "hat", "boat", "waves", "fork",
            "spoon", "bowl", "plate", "bread", "fish", "worm", "ant", "bee",
            "milk carton", "sun", "juice", "soda", "phone", "mouse", "chair",
            "pencil", "clock", "light", "book", "leaf", "wood", "glass",
            "shoe", "winter glove", "ring", "coin","rain", "fire", "bird",
            "sneakers", "moon", "volcano", "trashcan", "window",
            "kite", "wind", "snowman",
           ]
current_prompt = random.choice(prompts)

# Game timer
round_duration = 32
start_time = time.time()

# Brush settings
brushThickness = 15
drawColor = (0, 0, 255)  # red

xp, yp = 0, 0  # Previous finger position

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    # Detect hands
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:] # Middle finger tip

        fingers = detector.fingersUp()

        # Drawing Mode: Index finger up, middle down
        if fingers[1] == 1 and fingers[2] == 0:
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1
        else:
            xp, yp = 0, 0

    # Overlay canvas onto webcam
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Timer and prompt
    elapsed = int(time.time() - start_time)
    time_left = round_duration - elapsed

    cv2.putText(img, f"Prompt: {current_prompt}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(img, f"Time Left: {time_left}s", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    if time_left <= 0:
        cv2.putText(img, "TIME UP!", (400, 300),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 4)

        # Save final drawing
        savePath = "DrawingGame"
        os.makedirs(savePath, exist_ok=True)
        filename = os.path.join(savePath, f"{current_prompt}_{int(time.time())}.png")
        cv2.imwrite(filename, imgCanvas)
        break

    cv2.imshow("Game Mode", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
