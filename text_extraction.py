from PIL import Image
from yolo_extract import get_cur_and_other_user_messages, sort_messages_by_order


# def extract_text(image_input):

#     #The edited image is placed in composite. (Combination of layers + the base image)
#     #However, it has one extra layer in the shape (2340, 1080, 4), instead of (2340, 1080, 3), An alpha channel is added.
#     edited_image = image_input["composite"]
#     edited_image = cv2.cvtColor(edited_image, cv2.COLOR_RGBA2RGB)

#     print("test image composite ", edited_image.shape)
#     #TODO change this to model path of chat detector!
#     MODEL_PATH = "E:\Dating_LLM_Sentiment_Analysis/experimental_stuff/detection_folder/models/exported-models/det_model/saved_model"

#     image_np = dm.extract_chatbox(edited_image, model_path = MODEL_PATH)
#     reader = easyocr.Reader(['en'], gpu=True)
#     # text = reader.readtext(image_np)
#     text = reader.readtext(image_np)


#     # need to process text to get just the messages
#     return text

def add_role(role, msg_dictionary):
    for key in msg_dictionary:
        temp = msg_dictionary[key]
        temp.append(role)  # Append role to the existing list
        msg_dictionary[key] = temp  # Update the dictionary value
    return msg_dictionary



def extract_text(image_path):
    cur_user, other_user = get_cur_and_other_user_messages(image_path)


    cur_user = add_role("cur_user", cur_user)
    other_user = add_role("oth_user", other_user)

    all_user = cur_user

    all_user.update(other_user)
    all_dict = sort_messages_by_order(all_user)


    return all_dict