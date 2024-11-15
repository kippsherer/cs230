#!/usr/bin/env python

import sys
import numpy as np
import os
import cv2
import random
from glob import glob

# set some easily adjustable values
FPS = 30
FRAMEFACTOR = 2         # use every FRAMEFACTOR frames to create images
FRAMEMULTIPLIER = 2     # number of shifted frames created
ROTATIONMULTIPLIER = 4  # number of additional rotated frames per FRAMEMULTIPLIER
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
    files = os.listdir('D:\\storage\\Data\\MLdata\\dashcam\\positive')

    stats = {
        'files': 0,
        'img_orig': 0,
        'img_total': 0,
        'img_night': 0
    }

    for file in files:
        # make sure I have marked times, or print error and move to next file
        if file not in file_time_ranges:
            print("ERROR, no times defined for file: " + file)
            continue

        print("Extracting images from file: " + file)
        print("\t range: " + str(file_time_ranges[file]))

        # open the dashcam file
        video = cv2.VideoCapture('D:\\storage\\Data\\MLdata\\dashcam\\positive\\' + file)
        stats['files'] += 1
        #print("\t frames" + str( video.get(cv2.CAP_PROP_FRAME_COUNT)) )

        # skip to starting point for hazard
        frame_start = file_time_ranges[file][0] * FPS
        frame_end = file_time_ranges[file][1] * FPS
        print("\t frame range: " + str(frame_start) + " - " + str(frame_end))

        count = frame_start
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        # loop through frames 
        while video.isOpened() and count < frame_end:
            # grab a frame
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
                cv2.imwrite('D:\\storage\\Data\\MLdata\\dashcam\\frames_med\\pos_' + file[:-4] + 'frame' + str(count) + '_' + str(n) + '.png', resized_image)
                resized_image = cv2.flip(resized_image, 1)
                cv2.imwrite('D:\\storage\\Data\\MLdata\\dashcam\\frames_med\\pos_' + file[:-4] + 'frame' + str(count) + '_' + str(n) + 'f.png', resized_image)
                stats['img_total'] += 2
                if (file[21] == 'N'): stats['img_night'] += 2

                # scale and write an image to our 224 dataset, and a horizonal flipped version
                resized_image = cv2.resize(cropped_frame, (FRAME224, FRAME224), interpolation=cv2.INTER_AREA)
                cv2.imwrite('D:\\storage\\Data\\MLdata\\dashcam\\frames_224\\pos_' + file[:-4] + 'frame' + str(count) + '_' + str(n) + '.png', resized_image)
                resized_image = cv2.flip(resized_image, 1)
                cv2.imwrite('D:\\storage\\Data\\MLdata\\dashcam\\frames_224\\pos_' + file[:-4] + 'frame' + str(count) + '_' + str(n) + 'f.png', resized_image)

                # random rotation around 1920, FRAMEBOTTOM
                for r in range(ROTATIONMULTIPLIER):
                    d_adj = random.randint(-1*ROTATIONMAXDEGREES,ROTATIONMAXDEGREES)
                    rotation_matrix = cv2.getRotationMatrix2D((1920, FRAMEBOTTOM), d_adj, 1.0)
                    rotated_image = cv2.warpAffine(frame, rotation_matrix, (3840, 2160))

                    # scale and write an image to our med dataset, and a horizonal flipped version
                    cropped_frame = rotated_image[FRAMETOP+y_adj : FRAMETOP+y_adj+FRAMELARGE, FRAMELEFT+x_adj : FRAMELEFT+x_adj+FRAMELARGE]
                    resized_image = cv2.resize(cropped_frame, (FRAMEMED, FRAMEMED), interpolation=cv2.INTER_AREA)
                    cv2.imwrite('D:\\storage\\Data\\MLdata\\dashcam\\frames_med\\pos_' + file[:-4] + 'frame' + str(count) + '_' + str(n) + '_' + str(r) + '.png', resized_image)
                    resized_image = cv2.flip(resized_image, 1)
                    cv2.imwrite('D:\\storage\\Data\\MLdata\\dashcam\\frames_med\\pos_' + file[:-4] + 'frame' + str(count) + '_' + str(n) + '_' + str(r) + 'f.png', resized_image)
                    stats['img_total'] += 2
                    if (file[21] == 'N'): stats['img_night'] += 2

                    # scale and write an image to our 224 dataset, and a horizonal flipped version
                    resized_image = cv2.resize(cropped_frame, (FRAME224, FRAME224), interpolation=cv2.INTER_AREA)
                    cv2.imwrite('D:\\storage\\Data\\MLdata\\dashcam\\frames_224\\pos_' + file[:-4] + 'frame' + str(count) + '_' + str(n) + '_' + str(r) + '.png', resized_image)
                    resized_image = cv2.flip(resized_image, 1)
                    cv2.imwrite('D:\\storage\\Data\\MLdata\\dashcam\\frames_224\\pos_' + file[:-4] + 'frame' + str(count) + '_' + str(n) + '_' + str(r) + 'f.png', resized_image)

        print( stats )        
        video.release()
        cv2.destroyAllWindows()

    print( stats )


# easy place to list the start and stop times in the video file livestock is visible, in seconds
file_time_ranges = {
    '20241102145051_000040A.MP4': [47,99],
    '20241102145151_000041A.MP4': [0,8],
    '20241102145350_000043A.MP4': [6,18.25],
    '20241102145450_000044A.MP4': [55,99],
    '20241102145550_000045A.MP4': [0,22.25],
    '20241102192608_000117N.MP4': [34,35.5],
    '20241102192908_000120N.MP4': [22,31],
    '20241102193008_000121N.MP4': [28,55],
    '20241102193108_000122N.MP4': [15,22],
    '20241104171032_000074A.MP4': [36,99],  # better video
    '20241104171132_000075A.MP4': [0,33],    # better video
    '20241104172532_000089A.MP4': [25.5,28],
    '20241104172532_000089B.MP4': [38,53],
    '20241107111835_000109A.MP4': [46.5,47.5],
    '20241112161134_000724A.MP4': [55,58.5],
    '20241112161934_000732A.MP4': [47,51.5]
}
    




if __name__ == "__main__":
    main()