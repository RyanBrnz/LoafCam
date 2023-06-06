import numpy as np
import cv2 as cv


def importResizedImages(dir,div):
    img = cv.imread(dir, cv.IMREAD_COLOR)  # queryImage
    # Get the current width and height
    height, width = img.shape[:2]

    # Calculate the new width and height
    new_width = width // 3
    new_height = height // 3

    img = cv.resize(img, (new_width,new_height))
    
    return img

img1 = importResizedImages("TestImages\Desk1.jpg", 3) 
img2 = importResizedImages("TestImages\Desk2.jpg", 3) 

img1bw = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
img2bw = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

orb1 = cv.ORB_create(100)
orb2 = cv.ORB_create(100)

kp1 , des1 = orb1.detectAndCompute(img1bw,None)
kp2 , des2 = orb2.detectAndCompute(img2bw,None)

matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE)
matches = matcher.match(des1,des2)
#matches = sorted(matches, key = lambda X:X.distance)

img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:100], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


cv.imshow("corners", img3)
cv.waitKey(0)
cv.destroyAllWindows()

