from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from modules.db_interactions import initialise_db, get_documents
import replicate
import os


# model_path = "meta/meta-llama-3-8b-instruct"
model_path = "meta/meta-llama-3-70b-instruct"
# model_path = "meta/meta-llama-3.1-405b-instruct"
response = None
model = None
os.environ["REPLICATE_API_TOKEN"] = "r8_Jzt43L6OlVR9Z5E57qUk3IHTFxxlfed4JUPFN"


def AI_interest_eval(chat_history, collection):
    # We need to separate into multiple prompts. AI unable to handle such a big chunk of text.
    # 1) Grading based on the 3 categories, Capital Letters, Frequency and Emojis
    # 2) Giving suggestions + example messages to send.

    reference_chat = RAG(chat_history, collection)

    system_prompt = "You are a dating advisor."
    first_prompt = """
    The chat history is delimited by ###.
    Reference chat is delimited by !!!.
    Give the comments based on the chat history.
    Give suggestions with ideas from the reference chat. (but do not mention the existence of reference chat)
    The suggestions should create an engaging conversation that is flirty and asks the person out.

    1) Give an example message he should send
    2) Comment on how he can improve his messaging style
    3) Provide some interesting topics or pickup lines.
    4) Comment on the other user's interest level in the conversation

    ### {chat_history}
    !!! {reference_chat}
    """.format(chat_history=chat_history, reference_chat=reference_chat)

    AI_response_1 = replicate_generation(
        first_prompt, system_prompt)

    second_prompt = """
    The chat history is delimited by ###.
    Reference chat is delimited by !!!.

    Grade each user, cur_user and oth_user based on 3 sections: Capital Letters, Frequency of messages and Emojis.
    The score for Capital Letter is decided by the ratio of capital letters to all letters in the messages.\
    If there is a ratio of 0.5 capital letters, the score should be 5/10.

    The score for Emojis is bnased on the ratio of emojis to messages.\
    If there is an emoji in every message, the score is 10/10. If there is an emoji in only one of two messages, the score is 5/10.

    The score for Frequency of Messages is according to how many messages the user sends.\
    The higher the ratio of messages the user sends, the higher the score.\
    For example, if oth_user sends a ratio of 0.8 of all messages, his score will be a 8/10.

    This is a template of your response. Please edit the <insert_score> to appropriate values.
    -------------------------------------------
    cur_user:
    Capital letters: <insert_score>/10
    Emojis: <insert_score>/10

    Texting score: <insert_score>/20

    oth_user:
    Capital letters: <insert_score>/10
    Frequency: <insert_score>/10
    Emojis: <insert_score>/10

    Interest score: <insert_score>/30

    Comments for texting improvent on cur_user's part:
    For cur_user:
    1) Capital letters usage:
    2) Emojis usage:
    3) General texting comments:


    --------------------------------------------

    ### {chat_history}
    
    """.format(chat_history=chat_history)

    AI_response_2 = replicate_generation(
        second_prompt, system_prompt)

    AI_response = AI_response_1 + "\n" + AI_response_2

    print("AI_response: ", AI_response)

    return AI_response


def replicate_generation(prompt, system_prompt):

    print("Prompt sent to AI: ", prompt)

    output = replicate.run(
        model_path,
        input={
            "top_k": 50,
            "top_p": 0.9,
            "prompt": prompt,
            "ststem_prompt": system_prompt,
            "temperature": 0.3,
            "max_new_tokens": 1024,
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\
                <|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "presence_penalty": 0,
            "frequency_penalty": 0
        }
    )
    output = "".join(output)
    return output


def RAG(chat_history, collection):
    results = get_documents(collection, chat_history)
    print("this is RAG results: ", results)

    return results
