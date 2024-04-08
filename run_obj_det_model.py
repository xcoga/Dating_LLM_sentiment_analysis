import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import glob
import os

warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into TensorFlow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


def extract_bbox_region(image_np, boxes):
    """Extract regions from the image defined by bounding boxes.

    Args:
      image_np: numpy array representing the image
      boxes: array of bounding boxes in format [ymin, xmin, ymax, xmax]

    Returns:
      List of cropped regions of the image defined by the bounding boxes
    """
    cropped_regions = []
    img_height, img_width, _ = image_np.shape
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * img_width, xmax * img_width,
                                      ymin * img_height, ymax * img_height)
        cropped_regions.append(
            image_np[int(top):int(bottom), int(left):int(right)])
    return cropped_regions


def get_bounding_box_list(scores, detection_boxes, threshold=0.6):
    count = 0
    bounding_box_list = []

    # print("detection scores", scores)
    # print("detection boxes", detection_boxes)

    for score in scores:
        if score > threshold:
            bounding_box_list.append(detection_boxes[count])

    return bounding_box_list


def extract_chatbox(image, model_path):

    model = tf.saved_model.load(model_path)
    print("Model has been loaded!")

    # When we have more than 1 categories we have to refactor this
    # category_index = label_map_util.create_category_index_from_labelmap(
    #     path_to_labels, use_display_name=True)

    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    # The detection scores will show how likely it is for the box to be in the detected class. Hence, we can set a threshold to filter out highly likely boxes.
    bounding_box_list = get_bounding_box_list(
        detections['detection_scores'][0], detections['detection_boxes'][0], threshold=0.6)

    cropped_images = extract_bbox_region(image_np, bounding_box_list)


    print("croopped images", cropped_images)
    cropped_images = np.array(cropped_images)
    cropped_images = cropped_images[0]

    #returns images as list of numpy arrays
    return cropped_images



# def extract_chatbox(images_path, model_path, path_to_labels, output_dir):
#     count = 0
#     images = glob.glob(images_path)
#     model = tf.saved_model.load(model_path)
#     print("Model has been loaded!")

#     # When we have more than 1 categories we have to refactor this
#     # category_index = label_map_util.create_category_index_from_labelmap(
#     #     path_to_labels, use_display_name=True)

#     for image_path in images:
#         count += 1
#         print('Running inference for {}... '.format(image_path), end='')
#         image_np = load_image_into_numpy_array(image_path)
#         input_tensor = tf.convert_to_tensor(image_np)
#         # The model expects a batch of images, so add an axis with `tf.newaxis`.
#         input_tensor = input_tensor[tf.newaxis, ...]

#         detections = model(input_tensor)

#         # The detection scores will show how likely it is for the box to be in the detected class. Hence, we can set a threshold to filter out highly likely boxes.
#         bounding_box_list = get_bounding_box_list(
#             detections['detection_scores'][0], detections['detection_boxes'][0], threshold=0.6)

#         cropped_images = extract_bbox_region(image_np, bounding_box_list)

#         print("these are croped imgs, ", cropped_images)

#         # Saving the images from numpy into image
#         for i, cropped_image in enumerate(cropped_images):
#             output_path = os.path.join(
#                 output_dir, f"cropped_image_{count}_{i}.png")
#             Image.fromarray(cropped_image).save(output_path)

#         print('Done')


# if __name__ == '__main__':
#     IMAGE_PATH = "E:/Dating_LLM_Sentiment_Analysis/experimental_stuff/detection_folder/images/train/*.jpg"
#     MODEL_PATH = "E:\Dating_LLM_Sentiment_Analysis/experimental_stuff/detection_folder/models/exported-models/det_model/saved_model"
#     PATH_TO_LABELS = "E:/Dating_LLM_Sentiment_Analysis/experimental_stuff/detection_folder/annotations/label_map.pbtxt"
#     OUTPUT_DIR = "E:/Dating_LLM_Sentiment_Analysis/experimental_stuff/detection_folder/images/output_imgs"

#     extract_chatbox(IMAGE_PATH, MODEL_PATH, PATH_TO_LABELS, OUTPUT_DIR)
 