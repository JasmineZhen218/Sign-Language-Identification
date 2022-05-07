import cv2
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt
import math

def FrameCapture(pathin, pathout):
      
    # Path to video file
    vidObj = cv2.VideoCapture(pathin)
  
    # Used as counter variable
    count = 0
  
    # checks whether frames were extracted
    success, image = vidObj.read()
  
    while success:
  
        # vidObj object calls read
        # function extract frames
        

        # Saves the frames with frame-count
        #cv2.imwrite("/content/drive/MyDrive/ML_final/External_test_data/frame%d.jpg" % count, image)
        cv2.imwrite(pathout+"/%d.jpg" % count, image)
        image = rgb2gray(image)
        if count%5 == 0:
          
          plt.imshow(image, cmap='gray')
          plt.show()
        count += 1
        success, image = vidObj.read()

if __name__ == '__main__':
  FrameCapture("/content/drive/MyDrive/ML_final/IMG_0358.MP4", "/content/drive/MyDrive/ML_final/External_test_data")
