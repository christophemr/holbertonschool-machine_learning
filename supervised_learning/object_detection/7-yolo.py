import numpy as np
import tensorflow.keras as K
import cv2
import os


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
        with open(classes_path, 'r') as file:
            self.class_names = file.read().strip().split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _sigmoid(self, x):
        """Applies sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs of the Darknet model
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract box confidence, class probabilities, and coordinates
            box_confidence = self._sigmoid(output[..., 4:5])
            box_class_prob = self._sigmoid(output[..., 5:])
            box_xy = self._sigmoid(output[..., 0:2])  # Center coordinates
            box_wh = np.exp(output[..., 2:4])  # Width and height

            # Adjust the anchor boxes
            anchors = self.anchors[i].reshape((1, 1, len(self.anchors[i]), 2))
            box_wh *= anchors

            # Create grid to map coordinates
            # Here's where we make the first necessary change: using `.repeat(3, axis=-2)`
            col = np.tile(np.arange(0, grid_width), grid_height).reshape(-1, grid_width)
            row = np.tile(np.arange(0, grid_height), grid_width).reshape(-1, grid_width).T

            col = col.reshape(grid_height, grid_width, 1, 1).repeat(3, axis=-2)
            row = row.reshape(grid_height, grid_width, 1, 1).repeat(3, axis=-2)

            # Adjust coordinates
            box_xy += np.concatenate((col, row), axis=-1)
            box_xy /= (grid_width, grid_height)

            # Adjust width and height using model input shape, and convert center to corner coordinates
            box_wh /= (self.model.input.shape[1], self.model.input.shape[2])
            box_xy -= (box_wh / 2)  # Convert center xy to top-left corner

            boxes.append(np.concatenate((box_xy, box_xy + box_wh), axis=-1))

            # Store the confidences and class probabilities
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        # Adjust boxes to the original image size
        for i in range(len(boxes)):
            boxes[i][..., 0] *= image_width
            boxes[i][..., 1] *= image_height
            boxes[i][..., 2] *= image_width
            boxes[i][..., 3] *= image_height

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters the bounding boxes based on their confidence scores
        and class probabilities.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        # Iterate through each YOLO output (multiple grid sizes)
        for i in range(len(boxes)):
            # Reshape boxes, confidences, and class probabilities
            boxes_i = boxes[i].reshape(-1, 4)
            box_confidences_i = box_confidences[i].reshape(-1)
            box_class_probs_i = box_class_probs[i].reshape(-1, box_class_probs[i].shape[-1])

            # Multiply box confidences by class probabilities to get the box scores
            box_scores_i = box_confidences_i[:, np.newaxis] * box_class_probs_i

            # Get the highest score and corresponding class index for each box
            box_classes_i = np.argmax(box_scores_i, axis=-1)
            box_scores_i = np.max(box_scores_i, axis=-1)

            # Apply the class_t threshold to filter out low-scoring boxes
            mask = box_scores_i >= self.class_t

            filtered_boxes.append(boxes_i[mask])
            box_classes.append(box_classes_i[mask])
            box_scores.append(box_scores_i[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        # Debugging output to check the number of boxes after filtering
        print(f"Filtered {len(filtered_boxes)} boxes based on confidence threshold.")

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes.
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        # Set a lower IoU threshold to make the NMS more aggressive
        iou_threshold = 0.15

        for cls in unique_classes:
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_box_scores = box_scores[cls_mask]

            # Sort boxes by box score (higher score comes first)
            sorted_indices = np.argsort(cls_box_scores)[::-1]
            cls_boxes = cls_boxes[sorted_indices]
            cls_box_scores = cls_box_scores[sorted_indices]

            while len(cls_boxes) > 0:
                # Append the box with the highest score
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_box_scores[0])

                if len(cls_boxes) == 1:
                    break

                # Compute IoU between the top box and the rest
                iou = self._iou(cls_boxes[0], cls_boxes[1:])

                # Filter out boxes with IoU greater than or equal to the threshold
                keep_boxes = iou < iou_threshold

                # Debug: check how many boxes are being filtered out
                print(f"Boxes before filtering: {len(cls_boxes)}, Boxes after filtering: {np.sum(keep_boxes)}")

                # Keep only the boxes with IoU lower than the threshold
                cls_boxes = cls_boxes[1:][iou < self.nms_t]
                cls_box_scores = cls_box_scores[1:][iou < self.nms_t]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores


    def _iou(self, box1, boxes):
        """
        Calculate Intersection over Union (IoU) between box1 and other boxes.
        """
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union = box1_area + boxes_area - intersection
        iou = intersection / union

        # Debugging output to check IoU values
        print(f"IoU: {iou}, Box1: {box1}, Other Boxes: {boxes}")

        return iou


    @staticmethod
    def load_images(folder_path):
        """
        Loads images from a given folder path.
        """
        images = []
        image_paths = []

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"The directory {folder_path} does not exist")

        valid_extensions = ('.jpg', '.jpeg', '.png')

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)
                    image_paths.append(image_path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Resizes and rescales images for the Darknet model.
        """
        pimages = []
        image_shapes = []
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for img in images:
            # Resize image with inter-cubic interpolation
            resized_img = cv2.resize(
                img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)

            # Rescale pixel values from [0, 255] to [0, 1]
            pimages.append(resized_img / 255.0)

            # Add image shape to shapes array
            orig_h, orig_w = img.shape[:2]
            image_shapes.append([orig_h, orig_w])

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)
        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes, class names, and box scores
        """
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            class_idx = box_classes[i]
            score = box_scores[i]

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Prepare label text
            label = f"{self.class_names[class_idx]} {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Display the image
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            # Save the image in the 'detections' folder
            if not os.path.exists("detections"):
                os.makedirs("detections")
            save_path = os.path.join("detections", os.path.basename(file_name))
            cv2.imwrite(save_path, image)

        # Close the window
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Predict bounding boxes, class names, and scores for all images in a folder.

        Args:
            folder_path (str): Path to the folder containing the images to predict

        Returns:
            tuple: (predictions, image_paths)
                - predictions (list of tuples): Each tuple contains (boxes, box_classes, box_scores)
                - image_paths (list): A list of image paths corresponding to each prediction in predictions
        """
        predictions = []

        # Load images from the folder
        images, image_paths = self.load_images(folder_path)

        # Preprocess the images for the model
        preprocessed_images, image_shapes = self.preprocess_images(images)

        # Get the model outputs for each preprocessed image
        model_outputs = self.model.predict(preprocessed_images)

        # Iterate over each image and corresponding outputs
        for i, image in enumerate(images):
            outputs = [model_outputs[j][i] for j in range(len(model_outputs))]

            # Process the model outputs to get boxes, confidences, and class probabilities
            boxes, box_confidences, box_class_probs = self.process_outputs(outputs, image_shapes[i])

            # Filter the boxes using the confidence threshold
            filtered_boxes, box_classes, box_scores = self.filter_boxes(boxes, box_confidences, box_class_probs)

            # Apply Non-Maximum Suppression (NMS) to reduce overlapping boxes
            box_predictions, predicted_box_classes, predicted_box_scores = self.non_max_suppression(
                filtered_boxes, box_classes, box_scores)

            # Append the predictions for this image
            predictions.append((box_predictions, predicted_box_classes, predicted_box_scores))

            # Display the image with bounding boxes
            self.show_boxes(image, box_predictions, predicted_box_classes, predicted_box_scores, image_paths[i])

        return predictions, image_paths


