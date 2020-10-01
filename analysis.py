from read_data import Data
from scipy.optimize import fsolve
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import errno
import pandas as pd

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
        wells = np.asarray(wells)

        wells = np.append(np.zeros([len(wells), 2]), wells, axis = 1)

        # HoughCircles detects each of the wells seperately,
        # However, it is nacessary to label the wells based on their (x, y) coordinates in the images
        # So that they can be directly referenced later

        x_ref = wells[0, 2]; y_ref= wells[0, 3]; r_ref = wells[0, 4]
        x_ind = np.nonzero((wells[:, 2] > x_ref - r_ref) & (wells[:, 2] < x_ref + r_ref))[0]
        y_ind = np.nonzero((wells[:, 3] > y_ref - r_ref) & (wells[:, 3] < y_ref + r_ref))[0]

        median_x = []
        median_y = []

        for index in x_ind:
            median = np.median(wells[(wells[:, 3] > (wells[index, 3] - r_ref)) & (wells[:, 3] < (wells[index, 3] + r_ref)), 3])
            wells[(wells[:, 3] > (wells[index, 3] - r_ref)) & (wells[:, 3] < (wells[index, 3] + r_ref)), 1] = median
            median_y.append(median)

        for index in y_ind:
            median = np.median(wells[(wells[:, 2] > (wells[index, 2] - r_ref)) & (wells[:, 2] < (wells[index, 2] + r_ref)), 2])
            wells[(wells[:, 2] > (wells[index, 2] - r_ref)) & (wells[:, 2] < (wells[index, 2] + r_ref)), 0] = median
            median_x.append(median)

        wells = pd.DataFrame(wells, columns=['well_id_x', 'well_id_y', 'center_x', 'center_y', 'radius'])

        wells.replace({'well_id_x' : dict(zip(sorted(median_x), np.arange(0, len(median_x), 1, dtype = np.int16)))}, inplace = True)
        wells.replace({'well_id_y' : dict(zip(sorted(median_y), np.arange(0, len(median_y), 1, dtype = np.int16)))}, inplace = True)

        wells.set_index(['well_id_x', 'well_id_y'], inplace = True)

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

        wells = wells[['center_x', 'center_y', 'radius']].to_numpy()

        fig, ax = plt.subplots(figsize = (12, 12))
        ax.imshow(image)
        for circle in wells:
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

        print('Creating separate directories for each of the wells')

        well_no = 0
        for i in range(len(wells)):
            try:
                well_ind = np.unravel_index(well_no, (self.__Data.no_of_wells_x,
                                                            self.__Data.no_of_wells_y))
                os.makedirs(os.path.join(path, '{:02d}_{:02d}'.format(well_ind[0], well_ind[1])))
                well_no += 1
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise e

        print('created relevant directories')

        print ('Saving cropped images in {}'.format(path))

        if (full_dataset):
            self.__Data.reset()
            counter = 0
            counter2 = 0
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
                    filename = os.path.join(path, '{:02d}_{:02d}'.format(well_ind[0], well_ind[1]),
                                                'IMG_{:04d}.jpg'.format(counter))
                    cv2.imwrite(filename, cropped)
                    counter2 += 1
                    well_no += 1
        else:
            if image is None:
                raise ValueError("An image needs to provided to crop or set all = True")

            counter = 1
            well_no = 0
            counter2 = 0

            for well in wells:
                cropped = image[int(well[1]-well[2]):int(well[1]+well[2]),
                                    int(well[0]-well[2]):int(well[0]+well[2]), :]
                well_ind = np.unravel_index(well_no, (self.__Data.no_of_wells_x,
                                                            self.__Data.no_of_wells_y))
                filename = os.path.join(path, '{:02d}_{:02d}'.format(well_ind[0], well_ind[1]),
                                    'IMG_{:04d}.jpg'.format(counter))
                cv2.imwrite(filename, cropped)
                well_no += 1
                counter2 +=1

        print('Wrote {} files to {}'.format(counter2, path))
