import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import glob
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

def retrieve_detection_coordinates(detections):
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    return detections['num_detections'], detections['detection_classes']   


if __name__ == '__main__':
    IMAGE_PATH = "E:/tf_models/workspace/training_demo/images/train/*.jpg" 
    MODEL_PATH = "E:/tf_models/workspace/training_demo/exported-models/my_model/saved_model"
    PATH_TO_LABELS = "E:/tf_models/workspace/training_demo/annotations/label_map.pbtxt"
    OUTPUT_DIR = "E:/tf_models/workspace/training_demo/images/output_imgs"
    count = 0

     # Assuming all images are in JPEG format
    images = glob.glob(IMAGE_PATH)

    model = tf.saved_model.load(MODEL_PATH)
    print("Model has been loaded!")

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    for image_path in images:
        count +=1
        print('Running inference for {}... '.format(image_path), end='')
        image_np = load_image_into_numpy_array(image_path)
        # Model needs tensor inputs
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = model(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
        detections['num_detections'] = num_detections

        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        image_np_with_detections = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)




        plt.figure()
        plt.imshow(image_np_with_detections)
        plt.savefig(OUTPUT_DIR + f'/output_image_{count}.png') 
        plt.close()
        print('Done')