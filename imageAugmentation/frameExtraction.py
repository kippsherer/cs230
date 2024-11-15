#!/usr/bin/env python

import sys
import numpy as np
import os
import cv2
import random
from glob import glob

# set some easily adjustable values
FPS = 30
FRAMEFACTOR = 30        # use every FRAMEFACTOR frames to create images
FRAMEMULTIPLIER = 1     # number of additional shifted frames created
ROTATIONMULTIPLIER = 1  # number of additional rotated frames per FRAMEMULTIPLIER
ROTATIONMAXDEGREES = 12 # max tilt CW or CCW
FRAME224 = 224
FRAMEMED = FRAME224 * 2
FRAMELARGE = FRAME224 * 6
FRAMEAUGPIXELS = 100
FRAMEBOTTOM = 1820
FRAMETOP = FRAMEBOTTOM - FRAMELARGE - FRAMEAUGPIXELS # and exrta 50 to facilitate augmentation
FRAMELEFT = int( (3840 / 2) - (FRAMELARGE + FRAMEAUGPIXELS) / 2 )
FRAMERIGHT = int( FRAMELEFT + (FRAMELARGE + FRAMEAUGPIXELS) )


def main():
    # retrieve list of files with positive hazards
    files = os.listdir('D:\\storage\\Data\\MLdata\\dashcam')

    stats = {
        'files': 0,
        'img_orig': 0,
        'img_total': 0,
        'img_night': 0
    }

    for file in files:
        print("Extracting images from file: " + file)
        
        # open the dashcam file
        video = cv2.VideoCapture('D:\\storage\\Data\\MLdata\\dashcam\\' + file)
        stats['files'] += 1

        count = 0
        while video.isOpened():
            ret, frame = video.read()
            if ret == False:
                break

            # use every FRAMEFACTOR frame
            count += 1
            if count % FRAMEFACTOR:
                continue

            # loop FRAMEMULTIPLIER times choosing a random crop from within the boundries
            stats['img_orig'] += 1
            for n in range(FRAMEMULTIPLIER):
                # print(n)
                # find our random crop
                y_adj = random.randint(0,FRAMEAUGPIXELS)
                x_adj = random.randint(0,FRAMEAUGPIXELS)

                # scale and write an image to our med dataset, and a horizonal flipped version
                cropped_frame = frame[FRAMETOP+y_adj : FRAMETOP+y_adj+FRAMELARGE, FRAMELEFT+x_adj : FRAMELEFT+x_adj+FRAMELARGE]
                resized_image = cv2.resize(cropped_frame, (FRAMEMED, FRAMEMED), interpolation=cv2.INTER_AREA)
                cv2.imwrite('D:\\storage\\Data\\MLdata\\dashcam\\frames_med\\neg_' + file[:-4] + 'frame' + str(count) + '_' + str(n) + '.png', resized_image)
                resized_image = cv2.flip(resized_image, 1)
                cv2.imwrite('D:\\storage\\Data\\MLdata\\dashcam\\frames_med\\neg_' + file[:-4] + 'frame' + str(count) + '_' + str(n) + 'f.png', resized_image)
                stats['img_total'] += 2
                if (file[21] == 'N'): stats['img_night'] += 2

                # scale and write an image to our 224 dataset, and a horizonal flipped version
                resized_image = cv2.resize(cropped_frame, (FRAME224, FRAME224), interpolation=cv2.INTER_AREA)
                cv2.imwrite('D:\\storage\\Data\\MLdata\\dashcam\\frames_224\\neg_' + file[:-4] + 'frame' + str(count) + '_' + str(n) + '.png', resized_image)
                resized_image = cv2.flip(resized_image, 1)
                cv2.imwrite('D:\\storage\\Data\\MLdata\\dashcam\\frames_224\\neg_' + file[:-4] + 'frame' + str(count) + '_' + str(n) + 'f.png', resized_image)

                # random rotation around 1920, FRAMEBOTTOM
                for r in range(ROTATIONMULTIPLIER):
                    d_adj = random.randint(-1*ROTATIONMAXDEGREES,ROTATIONMAXDEGREES)
                    rotation_matrix = cv2.getRotationMatrix2D((1920, FRAMEBOTTOM), d_adj, 1.0)
                    rotated_image = cv2.warpAffine(frame, rotation_matrix, (3840, 2160))

                    # scale and write an image to our med dataset, and a horizonal flipped version
                    cropped_frame = rotated_image[FRAMETOP+y_adj : FRAMETOP+y_adj+FRAMELARGE, FRAMELEFT+x_adj : FRAMELEFT+x_adj+FRAMELARGE]
                    resized_image = cv2.resize(cropped_frame, (FRAMEMED, FRAMEMED), interpolation=cv2.INTER_AREA)
                    cv2.imwrite('D:\\storage\\Data\\MLdata\\dashcam\\frames_med\\neg_' + file[:-4] + 'frame' + str(count) + '_' + str(n) + '_' + str(r) + '.png', resized_image)
                    resized_image = cv2.flip(resized_image, 1)
                    cv2.imwrite('D:\\storage\\Data\\MLdata\\dashcam\\frames_med\\neg_' + file[:-4] + 'frame' + str(count) + '_' + str(n) + '_' + str(r) + 'f.png', resized_image)
                    stats['img_total'] += 2
                    if (file[21] == 'N'): stats['img_night'] += 2

                    # scale and write an image to our 224 dataset, and a horizonal flipped version
                    resized_image = cv2.resize(cropped_frame, (FRAME224, FRAME224), interpolation=cv2.INTER_AREA)
                    cv2.imwrite('D:\\storage\\Data\\MLdata\\dashcam\\frames_224\\neg_' + file[:-4] + 'frame' + str(count) + '_' + str(n) + '_' + str(r) + '.png', resized_image)
                    resized_image = cv2.flip(resized_image, 1)
                    cv2.imwrite('D:\\storage\\Data\\MLdata\\dashcam\\frames_224\\neg_' + file[:-4] + 'frame' + str(count) + '_' + str(n) + '_' + str(r) + 'f.png', resized_image)

        print( stats )        
        video.release()
        cv2.destroyAllWindows()
    print( stats )

if __name__ == "__main__":
    main()