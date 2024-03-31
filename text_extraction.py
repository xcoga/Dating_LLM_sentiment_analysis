from PIL import Image
import easyocr
import cv2
import numpy as np

def extract_text(image_input):
    #convert image to openCV formar
    cv_image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    #initialise reader and use it to read text from image
    reader = easyocr.Reader(['en'], gpu=True)
    text = reader.readtext(cv_image,paragraph=True)

    #need to process text to get just the messages
    return text