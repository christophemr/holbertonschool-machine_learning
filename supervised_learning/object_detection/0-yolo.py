#!/usr/bin/env python3
"""class Yolo that uses the Yolo v3 algorithm
to perform object detection
"""

import tensorflow.keras as k


class Yolo:
    """
    Class that uses Yolo v3 algorithm to perform object detection

    class constructor:
        def __init__(self, model_path, classes_path, class_t, nms_t, anchors)

    public instance attributes:
        model: the Darknet Keras model
        class_names: list of all the class names for the model
        class_t: the box score threshold for the initial filtering step
        nms_t: the IOU threshold for non-max suppression
        anchors: the anchor boxes
    """
    
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class for performing object detection
        using Yolo v3 algorithm.

        Parameters:
        - model_path: str, path to where a Darknet Keras model is stored.
        - classes_path: str, path to where the list of class names
        used for the Darknet model is stored.
        - class_t: float, representing the box score threshold for
        the initial filtering step.
        - nms_t: float, representing the IOU threshold for non-max suppression.
        - anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2),
        containing all of the anchor boxes.
        """
        self.model = k.models.load_model(model_path)
        self.class_names = self._load_class_names(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_class_names(self, classes_path):
        """
        Loads class names from the file specified in classes_path.

        Parameters:
        - classes_path: str, path to the file containing class names.

        Returns:
        - class_names: list of str, list of class names.
        """
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
