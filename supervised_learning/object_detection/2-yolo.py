#!/usr/bin/env python3
"""class Yolo that uses the Yolo v3 algorithm to perform object detection"""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Class that uses Yolo v3 algorithm to perform object detection

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
        """
        self.model = K.models.load_model(model_path)
        self.class_names = self._load_class_names(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_class_names(self, classes_path):
        """
        Loads class names from the file specified in classes_path.
        """
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet model outputs to extract boundary boxes,
        box confidences,
        and class probabilities.

        Parameters:
        - outputs: list of numpy.ndarrays containing the predictions from
          the Darknet model for a single image
        - image_size: numpy.ndarray containing the image’s original size
          [image_height, image_width]

        Returns:
        A tuple of (boxes, box_confidences, box_class_probs)
        """
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes = output.shape[:3]

            # Extract the box attributes t_x, t_y, t_w, t_h
            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            # Calculate the center (bx, by) and dimensions (bw, bh)
            bx = (
                self._sigmoid(tx) + np.arange(grid_width).reshape(
                    1, grid_width, 1)) / grid_width
            by = (
                self._sigmoid(ty) + np.arange(grid_height).reshape(
                    grid_height, 1, 1)) / grid_height
            bw = (
                np.exp(tw) * self.anchors[i][:, 0].reshape(
                    1, 1, anchor_boxes)) / self.model.input.shape[1]
            bh = (
                np.exp(th) * self.anchors[i][:, 1].reshape(
                    1, 1, anchor_boxes)) / self.model.input.shape[2]

            # Convert box (bx, by, bw, bh) to corner coordinates
            # (x1, y1, x2, y2)
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))

            # Extract the box confidence scores (objectness)
            box_confidence = self._sigmoid(output[..., 4:5])
            box_confidences.append(box_confidence)

            # Extract the class probabilities
            class_probs = self._sigmoid(output[..., 5:])
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    @staticmethod
    def _sigmoid(x):
        """ Sigmoid activation function """
        return 1 / (1 + np.exp(-x))

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters the bounding boxes based on their confidence scores
        and class probabilities.

        Args:
            boxes (list of numpy.ndarray):
            A list where each element is a numpy array
            of shape (grid_height, grid_width, anchor_boxes, 4)
            containing the bounding boxes for each output.
            box_confidences (list of numpy.ndarray):
            A list where each element is a numpy array
            of shape (grid_height, grid_width, anchor_boxes, 1)
            containing the box confidences.
            box_class_probs (list of numpy.ndarray):
            A list where each element is a numpy array
            of shape (grid_height, grid_width, anchor_boxes, classes)
            containing the class probabilities for each box.

        Returns:
            tuple:
                - filtered_boxes (numpy.ndarray): A numpy array of shape (?, 4)
                containing the filtered bounding boxes.
                - box_classes (numpy.ndarray): A numpy array of shape (?,)
                containing the predicted class index for each box.
                - box_scores (numpy.ndarray): A numpy array of shape (?)
                containing the confidence score for each box.
        """

        filtered_boxes = []
        box_classes = []
        box_scores = []

        # Iterate through each YOLO output (multiple grid sizes)
        for i in range(len(boxes)):
            # Reshape the boxes, confidences, and class probabilities
            boxes_i = boxes[i].reshape(-1, 4)
            box_confidences_i = box_confidences[i].reshape(-1)
            box_class_probs_i = box_class_probs[i].reshape(
                -1, box_class_probs[i].shape[-1])

            # Multiply confidences by class probabilities to get
            # the box scores for each class
            box_scores_i = box_confidences_i[:, np.newaxis] * box_class_probs_i

            # Get the max score and corresponding class index for each box
            # Class with the highest probability
            box_classes_i = np.argmax(box_scores_i, axis=-1)
            # Highest score across classes
            box_scores_i = np.max(box_scores_i, axis=-1)

            # Filter out boxes where the score is less than the threshold
            mask = box_scores_i >= self.class_t

            # Keep only boxes above the threshold
            filtered_boxes.append(boxes_i[mask])
            # Keep the corresponding class
            box_classes.append(box_classes_i[mask])
            # Keep the corresponding score
            box_scores.append(box_scores_i[mask])

        # Concatenate the results from all grid sizes
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
