import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

########################

brushThickness = 20
eraserThickness = 75
brushStyle = "normal"    # "normal" or "pixel"

########################

# Load Header Images
folderPath = "Header"
myList = os.listdir(folderPath)
print("Header files:", myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print("Total Headers:", len(overlayList))

header = overlayList[0]
drawColor = (0, 0, 175)

# Webcam
cap = cv2.VideoCapture(0)  # use 0 for default webcam
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # 1. Import image
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # --- Thumbs up → SAVE ARTWORK ---
        if fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            savePath = "SavedDrawings"
            os.makedirs(savePath, exist_ok=True)

            filename = os.path.join(savePath, f"Drawing_{int(time.time())}.png")
            cv2.imwrite(filename, imgCanvas)

            cv2.putText(img, f"Saved: {filename}", (400, 700),
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 3)

        # --- Peace sign → Pixel Brush ---
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
            brushStyle = "pixel"
            cv2.putText(img, "Pixel Brush Mode", (50, 650),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # --- Fist → Normal Brush ---
        elif all(f == 0 for f in fingers):
            brushStyle = "normal"
            cv2.putText(img, "Normal Brush Mode", (50, 650),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 4. Selection Mode: Two fingers up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # Purple rectangle when selecting
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25),
                          (153, 0, 76), cv2.FILLED)

            # Checking for click in header
            if y1 < 233:  # within header area
                if 0 < x1 < 160:
                    header = overlayList[0]
                    drawColor = (0, 0, 175)  # Red
                elif 160 < x1 < 320:
                    header = overlayList[1]
                    drawColor = (0, 165, 255)  # Orange
                elif 320 < x1 < 480:
                    header = overlayList[2]
                    drawColor = (100, 255, 255)  # Yellow
                elif 480 < x1 < 640:
                    header = overlayList[3]
                    drawColor = (115, 180, 60)  # Green
                elif 640 < x1 < 800:
                    header = overlayList[4]
                    drawColor = (255, 230, 205)  # Blue
                elif 800 < x1 < 960:
                    header = overlayList[5]
                    drawColor = (165, 95, 150)  # Purple
                elif 960 < x1 < 1120:
                    header = overlayList[6]
                    drawColor = (255, 200, 255)  # Pink
                elif 1120 < x1 < 1280:
                    header = overlayList[7]
                    drawColor = (0, 0, 0)  # Eraser

            cv2.rectangle(img, (x1, y1 - 25),
                          (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. Drawing Mode - Only index finger up
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 20, (153, 0, 76), cv2.FILLED)  # Purple dot            print("Drawing Mode")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if brushStyle == "normal":
                if drawColor == (0, 0, 0):  # Eraser
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:  # Normal brush
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            elif brushStyle == "pixel":
                if drawColor != (0, 0, 0):  # No pixel eraser
                    cv2.rectangle(img, (x1, y1), (x1 + 20, y1 + 20), drawColor, cv2.FILLED)
                    cv2.rectangle(imgCanvas, (x1, y1), (x1 + 20, y1 + 20), drawColor, cv2.FILLED)

            xp, yp = x1, y1


    # 6. Merge Canvas and Webcam feed
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # 7. Setting the header image
    img[0:233, 0:1280] = header

    # Display
    cv2.imshow("Image", img)

    # --- After merging canvas and webcam feed ---
    img[0:233, 0:1280] = header

    # Apply Style Transfer with keys
    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):  # Cartoon effect
        # Edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 9
        )
        # Smooth but preserve edges
        color = cv2.bilateralFilter(img, 9, 250, 250)
        # --- Vibrance boost ---
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, 40)  # increase saturation (+40)
        hsv = cv2.merge([h, s, v])
        vibrant = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Combine edges with vibrant color
        cartoon = cv2.bitwise_and(vibrant, vibrant, mask=edges)

        cv2.imshow("Styled", cartoon)

        savePath = "SavedDrawings"
        os.makedirs(savePath, exist_ok=True)
        filename = os.path.join(savePath, f"Drawing_Cartoon_{int(time.time())}.png")
        cv2.imwrite(filename, cartoon)


    elif key == ord('c'):  # Detail enhancement
        detail = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
        cv2.imshow("Styled", detail)
        cv2.imwrite("SavedDrawings/Drawing_Detail.png", detail)

    elif key == ord('q'):  # Quit
        break
