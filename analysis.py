import experiment_analysis import Data
from scipy.optimize import minimize
import cv2

class analysis:

    def __init__(self, Data_to_analyze):

        if not isinstance(Data_to_analyze, Data):
            raise TypeError('Data_to_analyze is not of type Data')

        self.__Data = Data

    def find_wells(self, image, minRadius = None, maxRadius = None):
        '''
            This method is used to automatically detect circular wells

            The method uses HoughCircles method to automatically detect the wells
            We use optimization methods to auto-detect the radius of the wells
            If auto detection doesn't work you can manually set limits for the radius

            NOTE: If you want to explicitly use a pre-defined minRadius and maxRadius
            values use the function detect_wells
        '''

        if image is None:
            ret, image = self.__iterator.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                return None
            # reset the video to its initial frame
            self.reset()

        if minRadius is None:
            minRadius, _, _ = image.shape
            minRadius = minRadius/(no_of_wells_x * 2)

        if maxRadius is None:
            maxRadius, _, _ = image.shape
            maxRadius = maxRadius/no_of_wells_x

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        R0 = [minRadius, maxRadius]

        R = minimize(self.__total_wells, R0, args = (image,), method='nelder-mead')

        wells = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 150, minRadius = R[0], maxRadius = R[1])

        print("Total number of wells detected = {}".format(len(wells[0])))

        return wells

    def detect_wells(self, Image, R):
        '''
            Detect all the wells which satisfy minRadius < well_radius < maxRadius
            using the HoughCircles method.
        '''

        if image is None:
            ret, image = self.__iterator.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                return None
            # reset the video to its initial frame
            self.reset()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        wells = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 150, minRadius = minRadius, maxRadius = maxRadius)

        return wells

    def __total_wells(self, R, image):

        wells = self.detect_wells(image, R)

        number_of_wells = len(wells)

        return abs((self.Data.number_of_wells_x * self.Data.number_of_wells_y) - number_of_wells)
