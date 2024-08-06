from PIL import Image
from .yolo_extract import get_cur_and_other_user_messages, sort_messages_by_order


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

    all_messages = []
    for key in all_dict:
        all_messages.append(all_dict[key][2]+": "+all_dict[key][0])
    
    all_messages = "\n".join(all_messages)
    
    return all_messages