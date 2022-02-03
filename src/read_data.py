import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Data:

    def __init__(self, filename):
        '''
            Load zebrafish images for analysis
            it can be:
                    name of video file (eg. video.avi)
                    or image sequence (eg. img_%02d.jpg, which will read samples like img_00.jpg, img_01.jpg, img_02.jpg, ...)

            input :
                filename : filename of a video or series of images as described above
        '''

        iterator = cv2.VideoCapture(filename)

        self.__iterator = iterator

    def reset(self):
        '''
            resets the video to its initial frame
        '''

        self.__iterator.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def read(self, plot = False):
        '''
            Get the next image

            ouput:
                ret : logical indicating if read was successful
                frame : next frame
        '''

        ret, frame = self.__iterator.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return ret, frame, 0

        if plot:
            fig, ax = plt.subplots(figsize = (12, 12))
            ax.imshow(frame)
            plt.show()

        frame_no = int(self.__iterator.get(cv2.CAP_PROP_POS_FRAMES))

        return ret, frame, frame_no

    def get_shape(self):
        '''
            Get the shape of images

            output :
                shape: shape of the images
        '''
        ret, frame, _ = self.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return None, None, None

        self.reset()

        return frame.shape

    def get_total_frames(self):
        '''
            Get total number of frames

            output :
                total frames in the imaging set
        '''
        return int(self.__iterator.get(cv2.CAP_PROP_FRAME_COUNT))