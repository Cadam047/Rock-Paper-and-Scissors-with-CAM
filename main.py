import random
import os
import numpy as np
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time
from PIL import Image, ImageDraw, ImageFont


# Load the Waltograph font
font_path = os.path.join("C:/Users/Ceje1/Desktop/FIU STUFF/FIU Fall 2024/Rock Paper and Scissor VS AI/Resources/Waltograph/waltographUI.ttf")
font = ImageFont.truetype(font_path, 100)  # Adjust font size as needed

# This code initialized the camera
cap = cv2.VideoCapture(0)

#Size of the BG
width = 1920
height = 1080

#Size of the camera
CameraWidth = 625
CameraHeight = 350

#width
cap.set(3, CameraWidth)
#height
cap.set(4, CameraHeight)

detector = HandDetector(maxHands=1)

timer = 0
stateResult = False
startGame = False
scores = [0,0] # [AI,PLAYER]

# This makes the camera work
while True:
    imgBG = cv2.imread("Resources/BGRPSGame.jpg")
    imgBG = cv2.resize(imgBG, (width, height))

    cv2.namedWindow("BG", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BG", width, height)

    success, img = cap.read()

    #imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
    #imgScaled = imgScaled[:,80: 480]

    img = cv2.resize(img, (CameraWidth, CameraHeight))

    x_offset = 1105
    y_offset = 390

    # Find Hands

    hands, img = detector.findHands(img) # with draw

    if startGame:

        if stateResult is False:
            timer = time.time() - initialTime
            # Create an image using PIL for custom text
            pil_img = Image.fromarray(cv2.cvtColor(imgBG, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            draw.text((910, 515), str(int(timer)), font=font, fill=(255, 255, 255))


            # Convert PIL image back to OpenCV format
            imgBG = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            if timer > 3:
                stateResult = True
                timer = 0
                
                if hands:
                    hand = hands[0]
                    playerMove = 0
                    fingers = detector.fingersUp(hand)
                    if fingers == [0,0,0,0,0]:
                        playerMove = 1
                    elif fingers == [1,1,1,1,1]:
                        playerMove = 2
                    elif fingers == [0,1,1,0,0]:
                        playerMove = 3
                    
                    images = {
    1: {
        'path': 'C:/Users/Ceje1/Desktop/FIU STUFF/FIU Fall 2024/Rock Paper and Scissor VS AI/Resources/RPSRock.png',
        'position': (285, 350)  # Position for RPSRock
    },
    2: {
        'path': 'C:/Users/Ceje1/Desktop/FIU STUFF/FIU Fall 2024/Rock Paper and Scissor VS AI/Resources/RPSPaper.png',
        'position': (215, 350)  # Position for RPSPaper
    },
    3: {
        'path': 'C:/Users/Ceje1/Desktop/FIU STUFF/FIU Fall 2024/Rock Paper and Scissor VS AI/Resources/RPSScissors.png',
        'position': (225, 450)  # Position for RPSScissors
    }
}
                    randomNumber = random.randint(1,3)
                    imgAI = cv2.imread(images[randomNumber]['path'], cv2.IMREAD_UNCHANGED)

                    
                    AIscale_percent = 35  # Adjust the scaling percentage as needed
                    AIwidth = int(imgAI.shape[1] * AIscale_percent / 100)
                    AIheight = int(imgAI.shape[0] * AIscale_percent / 100)
                    dim = (AIwidth, AIheight)
                    imgAI = cv2.resize(imgAI, dim, interpolation=cv2.INTER_AREA)
                    imgAI = cv2.flip(imgAI, 1)

                    position = images[randomNumber]['position']
                    imgBG = cvzone.overlayPNG(imgBG, imgAI, position)
                
                # Player Wins
                if (playerMove == 1 and randomNumber == 3) or \
                    (playerMove == 2 and randomNumber == 1) or \
                    (playerMove == 3 and randomNumber == 2):
                    scores[1] += 1
                
                # AI Wins
                if (randomNumber == 1 and playerMove == 3) or \
                    (randomNumber == 2 and playerMove == 1) or \
                    (randomNumber == 3 and playerMove == 2):
                    scores[0] += 1
                    print(playerMove)


    # Place the camera in the middle of the screen
    imgBG[y_offset:y_offset+CameraHeight, x_offset:x_offset+CameraWidth] = img

    if stateResult:
        imgBG = cvzone.overlayPNG(imgBG, imgAI, position)

    
    # This is the score for the player
    cv2.putText(imgBG, str(scores[0]), (665,340), cv2.FONT_HERSHEY_PLAIN,6,(255,255,255),4)


    # This is the score for the AI
    cv2.putText(imgBG, str(scores[1]), (1155,340), cv2.FONT_HERSHEY_PLAIN,6,(255,255,255),4)



    img = cv2.resize(img, (CameraWidth, CameraHeight))

    
    #cv2.imshow("Image", img)
    cv2.imshow("BG", imgBG)
    #cv2.imshow("Scaled", imgScaled)

    key = cv2.waitKey(1)
    if key == ord('s'):
        startGame = True
        initialTime = time.time()
        stateResult = False