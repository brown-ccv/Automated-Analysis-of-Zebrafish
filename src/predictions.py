import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow.compat.v1 as tf
from read_data import Data
from video_analysis import analysis
import pandas as pd
import numpy as np

class predict:

    def __init__(self, data, analysis, model_path):

        def load_graph(frozen_graph_filename):

            with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                with tf.Graph().as_default() as graph:
                    tf.import_graph_def(graph_def)
            return graph

        self.__analysis = analysis
        self.__data = data
        self.__graph = load_graph(model_path)

    def predict(self, wells, image = None):

        def predict_image(image, wells, sess, input, output):

            well_ind, data = self.__analysis.crop_wells(wells, image)
            predictions = sess.run(output, feed_dict = {input : data})
            
            predictions = np.asarray(predictions)[:,:2].reshape(len(wells), 6)

            predicted_image = pd.DataFrame(predictions, columns = ['right_eye_y', 'right_eye_x',
                                                                    'left_eye_y', 'left_eye_x',
                                                                    'yolk_y', 'yolk_x'], 
                                        index = pd.MultiIndex.from_tuples(well_ind))
    
            return predicted_image

        if (image):

            predicted_image = predict_image(image, wells, sess, input, output)
            sess.close()
            return predicted_image


        self.__data.reset()

        sess, output, input = self.get_session()
        predictions = []
        frames = []

        total_frames = self.__data.get_total_frames()

        for i in range(total_frames):
            ret, image, img_no = self.__data.read()

            frames.append(img_no)

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            predicted_image = predict_image(image, wells, sess, input, output)
            predictions.append(predicted_image)

            if (i%10) == 0:
                print ("Analyzed {}/{} images".format(i, total_frames), flush = True)

        predictions = pd.concat(predictions, keys = frames)
        predictions.rename_axis(['frame', 'X-coord', 'Y-coord'], inplace = True)

        sess.close()
        return predictions


    def get_session(self):
        sess = tf.Session(graph=self.__graph)
        output = sess.graph.get_tensor_by_name("import/concat_1:0")
        input = sess.graph.get_tensor_by_name("import/Placeholder:0")

        return sess, output, input
