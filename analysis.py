from read_data import Data
from scipy.optimize import fsolve
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class analysis:

    def __init__(self, Data_to_analyze):

        if not isinstance(Data_to_analyze, Data):
            raise TypeError('Data_to_analyze is not of type Data')

        self.__Data = Data_to_analyze

    def detect_wells( self, R, image = None ):
        '''
            Detect all the wells which satisfy minRadius < well_radius < maxRadius
            using the HoughCircles method.
        '''

        if image is None:
            ret, image = self.__Data.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                return None
            # reset the video to its initial frame
            self.__Data.reset()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        wells = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 150, minRadius = int(R[0]), maxRadius = int(R[1]))
        wells = sorted(wells[0], key = lambda x: (x[0], x[1]))

        return wells

    def plot_wells(self, wells, image = None):
        '''
            Once you've detected the wells you can plot them using this function
        '''

        if image is None:
            ret, image = self.__Data.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                return None
            # reset the video to its initial frame
            self.__Data.reset()

        npc = np.asarray(wells, dtype=np.float32)
        fig, ax = plt.subplots(figsize = (12, 12))
        ax.imshow(image)
        for circle in npc:
            ax.add_artist(plt.Circle((circle[0], circle[1]), circle[2]))

    def crop_wells(self, wells, full_dataset = True, image = None, crop_dir = None):
        '''
            Crop each of the wells into single images and write them to files
        '''

        if not crop_dir:
            path = os.path.join(os.getcwd(), 'cropped_images')
        else:
            path = crop_dir

        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e

        if (full_dataset):
            self.__Data.reset()
            counter = 0
            while (True):
                counter += 1
                well_no = 0
                ret, image = self.__Data.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                for well in wells:
                    cropped = image[int(well[1]-well[2]):int(well[1]+well[2]),
                                        int(well[0]-well[2]):int(well[0]+well[2]), :]
                    well_ind = np.unravel_index(well_no, (self.__Data.no_of_wells_x,
                                                                self.__Data.no_of_wells_y))
                    filename = os.path.join(path,
                            'IMG_{:04d}_{:02d}_{:02d}.jpg'.format(counter, well_ind[0], well_ind[1]))
                    cv2.imwrite(filename, cropped, params = 'JPEG')
                    well_no += 1
        else:
            if image is None:
                raise ValueError("An image needs to provided to crop or set all = True")

            counter = 1
            well_no = 0

            for well in wells:
                cropped = image[int(well[1]-well[2]):int(well[1]+well[2]),
                                    int(well[0]-well[2]):int(well[0]+well[2]), :]
                well_ind = np.unravel_index(well_no, (self.__Data.no_of_wells_x,
                                                            self.__Data.no_of_wells_y))
                filename = os.path.join(path,
                        'IMG_{:04d}_{:02d}_{:02d}.jpg'.format(counter, well_ind[0], well_ind[1]))
                cv2.imwrite(filename, cropped)
                well_no += 1
