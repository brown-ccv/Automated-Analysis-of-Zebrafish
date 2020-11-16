from read_data import Data
from scipy.optimize import fsolve
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import errno
import pandas as pd
import random

class analysis:

    def __init__(self, Data_to_analyze):
        '''
            This class sets up pre-processing routines to be fed into deeplabcut
            or tensorflow for training or inference

            input :
                Data_to_analyze : A read_data class with loaded data
        '''

        if not isinstance(Data_to_analyze, Data):
            raise TypeError('Data_to_analyze is not of type Data')

        self.__Data = Data_to_analyze

    def detect_wells( self, R, image = None ):
        '''
            Detect all the wells which satisfy minRadius < well_radius < maxRadius
            using the HoughCircles method.

            input :
                R = [minRadius, maxRadius]
                image = specific image where the wells need to be detected
                        if (None) : take the first image from the folder
            output :
                wells : Pandas dataframe indicating well locations
                        DataFrame Discription :
                            columns = { 'well_id_x' - x-coordinate of the well,
                                        'well_id_y' - y-coordinate of the well,
                                        'center_x' - x-coordinate of the center pixel of well,
                                        'center_y' - y-coordinate of the center pixel of well,
                                        'radius' - radius of the well}
        '''


        # initialize the image for well detection
        if image is None:
            ret, image = self.__Data.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                return None
            # reset the video to its initial frame
            self.__Data.reset()

        # workflow for detecting wells
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        wells = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 150, minRadius = int(R[0]), maxRadius = int(R[1]))
        wells = sorted(wells[0], key = lambda x: (x[0], x[1]))
        wells = np.asarray(wells)

        wells = np.append(np.zeros([len(wells), 2]), wells, axis = 1)

        # HoughCircles detects each of the wells seperately,
        # However, it is nacessary to label the wells based on their (x, y) coordinates in the images
        # So that they can be directly referenced later

        return self.__label_wells(wells)

    def plot_wells(self, wells, image = None):
        '''
            Once you've detected the wells you can plot them using this function

            input :
                wells : pandas dictionary of detected wells
                image : specific image on which wells are to be plotted
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


    def crop_to_video(self, wells, crop_dir = None, no_wells_to_record = 6):
        '''
            Crop each of the wells into single images and write them as a video

            input :
                wells : pandas dictionary of detected wells
                crop_dir : locations where the videos are to be stored

            output :
                filenames : locations of all filenames
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

        well_no = 0
        VideoWriter = {}
        height, width, channels = self.__Data.get_shape()
        size = (height, width)

        # create videowriter elements foreach of the wells
        filenames = []
        random_wells = random.sample(wells.index.values.tolist(), no_wells_to_record)
        for well_ind in random_wells:
            filename = os.path.join(path,
                                    '{:02d}_{:02d}.avi'.format(int(well_ind[0]), int(well_ind[1])))
            filenames.append(filename)
            VideoWriter[well_ind] = cv2.VideoWriter(filename,
                                                    cv2.VideoWriter_fourcc(*'DIVX'),
                                                    1,
                                                    (2*int(wells.loc[well_ind]['radius']),
                                                    2*int(wells.loc[well_ind]['radius'])))

        print ('Saving cropped images in {} as a videos'.format(path))

        self.__Data.reset()
        while (True):
            ret, image = self.__Data.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            for well_ind in random_wells:
                xc, yc, r = wells.loc[well_ind]
                cropped = image[int(yc)-int(r):int(yc)+int(r), int(xc)-int(r):int(xc)+int(r), :]
                VideoWriter[well_ind].write(cropped)

        for _, w in VideoWriter.items():
            w.release()

        print('Wrote {} videos to {}'.format(len(VideoWriter), path))

        self.__Data.reset()

        return filenames

    def crop_wells(self, wells, image):
        '''
            crop each image and return a numpy array

            input :
                wells : pandas dictionary of detected wells
                image : image to be cropped

            output :
                cropped_wells : a numpy array of cropped images
        '''

        cropped_images = np.full((len(wells.index), 2*int(wells.loc[(0, 0)]['radius']),
                                        2*int(wells.loc[(0, 0)]['radius']), 3), 0)

        for i, well_ind in enumerate(wells.index.values.tolist()):
            xc, yc, r = wells.loc[well_ind]
            cropped = image[int(yc)-int(r):int(yc)+int(r), int(xc)-int(r):int(xc)+int(r), :]
            cropped_images[i, :, :, :] = cropped

        return wells.index.values.tolist(), cropped_images

    def __label_wells(self, wells):
        '''
            Label the (x, y) coordinates for each well
        '''


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
        wells['radius'] = np.ceil(wells['radius'].median())

        wells.set_index(['well_id_x', 'well_id_y'], inplace = True)

        return wells
