from PIL import Image
import easyocr
import cv2
import numpy as np
import run_obj_det_model as dm


def extract_text(image_input):

    #The edited image is placed in composite. (Combination of layers + the base image)
    #However, it has one extra layer in the shape (2340, 1080, 4), instead of (2340, 1080, 3), An alpha channel is added.
    edited_image = image_input["composite"]
    edited_image = cv2.cvtColor(edited_image, cv2.COLOR_RGBA2RGB)

    print("test image composite ", edited_image.shape)
    #TODO change this to model path of chat detector!
    MODEL_PATH = "E:\Dating_LLM_Sentiment_Analysis/experimental_stuff/detection_folder/models/exported-models/det_model/saved_model"

    image_np = dm.extract_chatbox(edited_image, model_path = MODEL_PATH)
    reader = easyocr.Reader(['en'], gpu=True)
    # text = reader.readtext(image_np)
    text = reader.readtext(image_np)


    # need to process text to get just the messages
    return text


