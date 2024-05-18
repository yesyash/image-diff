# Import packages
import os
import cv2
import imutils
import numpy as np

def findDiffAndWriteToFile(image1: str, image2: str):
    diffImagesFolderName = "diff"

    if not os.path.exists(diffImagesFolderName):
        os.makedirs(diffImagesFolderName)

    # Load the two images
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    # Get the filenames without extension
    filename1 = os.path.splitext(os.path.basename(image1))[0]
    filename2 = os.path.splitext(os.path.basename(image2))[0]

    img1_height = img1.shape[0]
    img1_width = img1.shape[1]

    img2_height = img2.shape[0]
    img2_width = img2.shape[1]

    # Find the min dimensions of the two images
    img_height = min(img1_height, img2_height)
    img_width = min(img1_width, img2_width)

    # Resize the images to match exactly match the dimensions
    img1 = cv2.resize(img1, (img_width, img_height))
    img2 = cv2.resize(img2, (img_width, img_height))

    # Grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find the difference between the two images
    # Calculate absolute difference between two arrays 
    diff = cv2.absdiff(gray1, gray2)
    cv2.imshow("diff(img1, img2)", diff)

    # Apply threshold. Apply both THRESH_BINARY and THRESH_OTSU
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Threshold", thresh)

    # Dilation
    kernel = np.ones((5,5), np.uint8) 
    dilate = cv2.dilate(thresh, kernel, iterations=2) 
    cv2.imshow("Dilate", dilate)

    # Calculate contours
    contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            # Calculate bounding box around contour
            x, y, w, h = cv2.boundingRect(contour)
            # Draw rectangle - bounding box on both images
            cv2.rectangle(img1, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.rectangle(img2, (x, y), (x+w, y+h), (0,0,255), 2)

    # Show images with rectangles on differences
    x = np.zeros((img_height,10,3), np.uint8)
    result = np.hstack((img1, x, img2))

    cv2.imwrite(f"{diffImagesFolderName}/{filename1}-{filename2}.jpg", img=result, params=[cv2.IMWRITE_JPEG_QUALITY, 100])


images = [['images/plane1.png', 'images/plane2.png'], ['images/city1.jpg', 'images/city2.jpg']]

for index, image in enumerate(images):
    findDiffAndWriteToFile(image1=image[0], image2=image[1])
