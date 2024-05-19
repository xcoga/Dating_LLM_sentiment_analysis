from PIL import Image
from ultralytics import YOLO
import easyocr
import numpy as np
import os
import re
import shutil

############################# functions processing the image  ################################################################
#This function processes the YOLO readings and retrieve which coordinates belong to which class.
def extract_message_components(box_coordinates, classes):

    chatframe = []
    cur_user_boxes = []
    oth_user_boxes = []
    message_statuses = []
    timestamps = []

    count = 0

    for val in classes:
        #If class is chatframe
        if val == 0:
            chatframe.append(box_coordinates[count])
        #If the class is current_user
        elif val == 1:
            cur_user_boxes.append(box_coordinates[count])
        #If the class is other user
        elif val == 2:
            oth_user_boxes.append(box_coordinates[count])
        #If class is message status
        elif val == 3:
            message_statuses.append(box_coordinates[count])
        #If class is timestamp
        else:
            timestamps.append(box_coordinates[count])

        count+=1

    return chatframe, cur_user_boxes, oth_user_boxes, message_statuses, timestamps

#This function checks if coord1 is within coord2
def is_within(coord1, coord2):
    # Check if coord1 (x1, y1, x2, y2) is within coord2 (x1, y1, x2, y2)
    x1, y1, x2, y2 = coord1
    xm1, ym1, xm2, ym2 = coord2
    return x1 >= xm1 and y1 >= ym1 and x2 <= xm2 and y2 <= ym2


#This function will combine the arguments to form a list of elements of [Message, status, Timestamp] for each user.
def combine_message_components(cur_user_msgs, oth_user_msgs, message_statuses, timestamps):

    #Format of list to be [Message, status, Timestamp]

    combined_cur_user = []
    combined_oth_user = []


    for val in cur_user_msgs:
        combined_cur_user.append([val,None,None])
    
    for val in oth_user_msgs:
        combined_oth_user.append([val,None,None])

    for status in message_statuses:
        for i in range(len(combined_cur_user)):
            #If the status coordinates is within the message box coordinates
            if is_within(status, combined_cur_user[i][0]):
                temp = combined_cur_user[i]
                temp[1] = status
                combined_cur_user[i] = temp

        for i in range(len(combined_oth_user)):
            #If status coordinates is within the message box coordinates
            if is_within(status, combined_oth_user[i][0]):
                temp = combined_oth_user[i]
                temp[1] = status
                combined_oth_user[i] = temp

    
    for ts in timestamps:
        for i in range(len(combined_cur_user)):
            #If the status coordinates is within the message box coordinates
            if is_within(ts, combined_cur_user[i][0]):
                temp = combined_cur_user[i]
                temp[2] = ts
                combined_cur_user[i] = temp

        for i in range(len(combined_oth_user)):
            #If status coordinates is within the message box coordinates
            if is_within(ts, combined_oth_user[i][0]):
                temp = combined_oth_user[i]
                temp[2] = ts
                combined_oth_user[i] = temp

    return combined_cur_user, combined_oth_user


#This function is to split the readings from the YOLO model into the 2 users involved in the conversation.
def get_messages_components_list(results):
    for r in results:
        boxes = r.boxes

        classes = boxes.cls
        classes = classes.tolist()

        #There are multiple formats for the output of the xy coordinates, but we will use xyxy
        coordinates = boxes.xyxy
        coordinates = coordinates.tolist()

        chatframe, cur_user_boxes, oth_user_boxes, message_statuses, timestamps = extract_message_components(coordinates, classes)
        cur_user_msgs, oth_user_msgs = combine_message_components(cur_user_boxes, oth_user_boxes, message_statuses, timestamps)

    return cur_user_msgs, oth_user_msgs


#This function is to completely blacken parts of the image that we do not want.
def remove_image_parts(image, xyxy_coordinates):
    

    #Because of the image_np requirements, lets round the xyxy_coordinates to nearest int
    xyxy_coordinates = [round(coord) for coord in xyxy_coordinates]


    image_np = np.array(image)

    x1 = xyxy_coordinates[0]
    y1 = xyxy_coordinates[1]
    x2 = xyxy_coordinates[2]
    y2 = xyxy_coordinates[3]
    

    ##Saving this part before its deleted in case necessary 
    removed_part = image.crop((x1,y1,x2,y2))


    image_np[y1:y2, x1:x2] = (0,0,0)
    img = Image.fromarray(image_np)

    return img, removed_part

#This function is to crop out the status and timestamps before feeding to easyOCR, to get better readings.
def crop_image_parts(msg_components_list, image_path, save_path = "/home/Dating_LLM_sentiment_analysis/Cropping"):
    img = Image.open(image_path).convert('RGB') 
    
    for val in msg_components_list:
        img_copy = img
        msg = val[0]
        status = val[1]
        timestamp = val[2]

        #First, remove the timestamps and statuses, as they affect the EasyOCR recognition.
        if val[1] != None:
            img_copy, status_img = remove_image_parts(img_copy, val[1])
        if val[2] != None:
            img_copy, timestamp_img = remove_image_parts(img_copy, val[2])



        #Since we ar eusing the timestamp and the message images, we save them. Discard the status since its unused.
        timestamp_img.save(f"{save_path}/ts_{val[0]}.jpg")
        #Crop the message from the image
        msg = img_copy.crop(msg)
        msg.save(f"{save_path}/msg_{val[0]}.jpg")    

#This function is an auxiliary function used to clear out the entire folder
def clear_folder(folder_path):
    try:

        # Delete the folder and all its contents
        shutil.rmtree(folder_path)
        
        # Recreate the folder
        os.makedirs(folder_path)
        
        print(f"Folder '{folder_path}' deleted and recreated successfully.")
    except Exception as e:
        print(f"Failed to delete and recreate folder '{folder_path}'. Reason: {e}")





#################################### functions after processing the image ####################################
#This function runs and returns the results of the YOLO detection
def run_YOLO_model(model_path, image_path):
    model = YOLO(model_path)

    return model(image_path)

#This function will run the model on all images in the Cropping folder and return its results in a list.
def run_easy_OCR(folder_path):
    result_list = []
    appended_text = ""

    reader = easyocr.Reader(['en'])
    files = os.listdir(folder_path)
    
    #sort files to have message come before timestamp.
    #This is to sort the result_list before it gets into a bigger mess to sort.
    files = sorted(files, key=custom_sort_key)

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        appended_text = ""
        
        if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
            try:
                # Open the image using PIL (Python Imaging Library)
                img = Image.open(file_path)

                # Perform OCR on the image
                result = reader.readtext(img)

                # Output the OCR result
                print(f"=== OCR Result for {file_name} ===")
                #TODO this is the bug. If there are multiple lines, this will not work.
                for i in range(len(result)):
                    #To make it neater to read, for messages, we add a new line and spacing in the string.
                    if file_name.startswith('msg'):
                        appended_text += "\n " + result[i][1]
                    else:
                        appended_text += result[i][1]

                print("final appended text: ", appended_text)
                result_list.append((file_name, appended_text))
                print("\n")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    #Clear the folder, so that wont mix up with next image

    clear_folder(folder_path)


    return result_list

#This function is an auxiliary function for 'run_easy_OCR' in order to sort the files in the order of their starting name: 'msg, ts, other files'
def custom_sort_key(filename):
    if filename.startswith('msg'):
        # Filenames starting with 'msg' should come first
        return (0, filename)  # Return tuple (0, filename)
    elif filename.startswith('ts'):
        # Filenames starting with 'ts' should come after 'msg'
        return (1, filename)  # Return tuple (1, filename)
    else:
        # Other filenames not starting with 'msg' or 'ts'
        return (2, filename)  # Return tuple (2, filename)


#This function is to place together the message and time in the results_list under a single key in the dictionary.
def pair_time_and_message(result_list):
    paired_dict = {}

    for val in result_list:
        key = extract_filename(val[0])  # Extract key from the filename

        if key not in paired_dict:
            #If there was a failure detecting the message, and only the timestamp is detected, then abandon the timestamp
            if val[0].startswith('ts'):
                continue
            paired_dict[key] = [val[1]]  # Initialize an empty list for the key if not present

        else:
            temp = paired_dict[key]
            temp.append(val[1])
            paired_dict[key] = temp

    return sort_messages_by_order(paired_dict)



#This function is auxiliary fn for 'pair_time_and_message'. This is used to retrieve [1,2,3,4] coordinates from filename 'ts_[1,2,3,4].jpg'
def extract_filename(filename):
    match = re.search(r'\[(.*?)\]', filename)
    if match:
        values_str = match.group(1)  # Get the string inside brackets
        return values_str
    else:
        return None 
        

#This function will sort the messages in the dictionary 'data' in the correct order, based on their y-coordinate on the screen
def sort_messages_by_order(data):
    # Parse and sort keys based on y-axis values
    sorted_keys = sorted(
        ((
            min(float(key.split(',')[1]), float(key.split(',')[3])),  # y1
            max(float(key.split(',')[1]), float(key.split(',')[3])),  # y2
            key
        ) for key in data),
        key=lambda item: item[0]  # Sort based on y1 (you can use item[1] for sorting based on y2)
    )

    # Rebuild the dictionary with sorted keys
    sorted_dict = {item[2]: data[item[2]] for item in sorted_keys}

    # Output the sorted dictionary
    print(sorted_dict)
    return sorted_dict


def get_cur_and_other_user_messages(img_path):

    YOLO_path = "./models/best.onnx"
    cropped_img_path_cur_user = "./Cropping/cur_user"
    cropped_img_path_oth_user = "./Cropping/oth_user"

    results = run_YOLO_model(YOLO_path, img_path)
    cur_user_msgs, oth_user_msgs = get_messages_components_list(results)
    crop_image_parts(cur_user_msgs, img_path, cropped_img_path_cur_user)
    crop_image_parts(oth_user_msgs, img_path, cropped_img_path_oth_user)

    prediction_list_cur_user = run_easy_OCR(cropped_img_path_cur_user)
    prediction_list_oth_user = run_easy_OCR(cropped_img_path_oth_user)

    paired_dict_cur_user = pair_time_and_message(prediction_list_cur_user)
    paired_dict_oth_user = pair_time_and_message(prediction_list_oth_user)


    return paired_dict_cur_user, paired_dict_oth_user

# if __name__ == "__main__":

#     img_path = "/home/data/images/train/Screenshot_20240519_110552_Telegram.jpg"
#     YOLO_path = "/home/best.onnx"
#     cropped_img_path_cur_user = "/home/Cropping/cur_user"
#     cropped_img_path_oth_user = "/home/Cropping/oth_user"

#     results = run_YOLO_model(YOLO_path, img_path)
#     cur_user_msgs, oth_user_msgs = get_messages_components_list(results)
#     crop_image_parts(cur_user_msgs, img_path, cropped_img_path_cur_user)
#     crop_image_parts(oth_user_msgs, img_path, cropped_img_path_oth_user)

#     prediction_list_cur_user = run_easy_OCR(cropped_img_path_cur_user)
#     prediction_list_oth_user = run_easy_OCR(cropped_img_path_oth_user)

#     paired_dict_cur_user = pair_time_and_message(prediction_list_cur_user)
#     paired_dict_oth_user = pair_time_and_message(prediction_list_oth_user)


















    


