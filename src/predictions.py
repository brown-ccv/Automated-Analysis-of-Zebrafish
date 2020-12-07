import tensorflow as tf
from read_data import Data
from analysis import analysis
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

            predictions = np.asarray(predictions).reshape(3, len(wells), 3)
            predictions = (predictions.view([(f'f{i}',predictions.dtype)
                                            for i in range(predictions.shape[-1])])[...,0].astype('O'))

            predicted_image = pd.DataFrame( predictions.transpose(), 
                                            columns = ['right_eye', 'left_eye', 'yolk'],
                                            index = pd.MultiIndex.from_tuples(well_ind))

            return predicted_image

        if (image):

            predicted_image = predict_image(image, wells, sess, input, output)
            sess.close()
            return predicted_image


        self.__data.reset()

        sess, output, input = self.get_session()
        predictions = []

        while (True):
            ret, image, img_no = self.__data.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            predicted_image = predict_image(image, wells, sess, input, output)
            predictions.append(predicted_image)

        predictions = pd.concat(predictions, keys = [i + 1 for i in range(self.__data.get_total_frames())])

        sess.close()
        return predictions


    def get_session(self):
        sess = tf.Session(graph=self.__graph)
        output = sess.graph.get_tensor_by_name("import/concat_1:0")
        input = sess.graph.get_tensor_by_name("import/Placeholder:0")

        return sess, output, input
