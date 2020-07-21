"""
Extract images from video.

Ex. For FPS & AP analysis, select a video to perform analysis on. Then extract images with this code.
"""
# Importing all necessary libraries
import cv2
import os

VIDEO_IMAGE_PATH = 'video_images'  # dir where images extracted from video should be saved

# Read the video from specified path
cam = cv2.VideoCapture("2019_07_24_1_Up_Crash.MP4")

try:

    # creating a folder named data
    if not os.path.exists('video_images'):
        os.makedirs('video_images')

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0

print('Starting image extraction.')
while True:

    # reading from frame
    ret, frame = cam.read()

    if ret:
        # if video is still left continue creating images
        name = './' + VIDEO_IMAGE_PATH + '/frame' + str(currentframe) + '.jpg'
        # print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
print('Finished extracting images from video.')