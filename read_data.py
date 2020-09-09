import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

class Data:

    def __init__(self, filename, no_of_wells_x, no_of_wells_y):
        '''
            Load zebrafish images for analysis
            it can be:
                    name of video file (eg. video.avi)
                    or image sequence (eg. img_%04d.jpg, which will read samples like img_00.jpg, img_01.jpg, img_02.jpg, ...)
        '''

        iterator = cv2.VideoCapture(filename)

        self.__iterator = iterator
        self.no_of_wells_x = no_of_wells_x
        self.no_of_wells_y = no_of_wells_y

    def reset(self):
        '''
            resets the video to its initial frame
        '''

        self.__iterator.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def read(self):
        '''
            Get the next images
        '''
        ret, frame = self.__iterator.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return None

        return frame


    def show(self):
        '''
            Display the series of images or the video
        '''

        while (True):
            # read video frame-by-frame
            ret, frame = self.__iterator.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame = cv2.resize(frame, (960, 960))
            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()

        # reset the video to its initial frame
        self.reset()

    def find_wells(self, image = None):
        '''
            Detect wells using the hough_circles algorithm

            image = image for well detection (if None first image of the sequence is used)
            number_of_wells = Number of wells per plate
        '''

        if image is None:
            ret, image = self.__iterator.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                return None
            # reset the video to its initial frame
            self.reset()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        wells = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 150, minRadius = 70, maxRadius = 108)

        print("Total number of wells detected = {}".format(len(wells[0])))

        return wells

    # def background_subtraction(method):
    #
    #     fgbg = method()
    #
    #     while (True):
    #
    #         ret, frame = Data._iterator.read()
    #         if not read:
    #             print("Can't receive frame (stream end?). Exiting ...")
    #             break
    #
    #         fmask = fgbg.apply(frame)
    #
    # def adaptive_threshholding(method, threshhold_type):
    #
    #     fmask =
