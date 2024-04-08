import gradio as gr
from dotenv import load_dotenv
import random
import os
import time
import replicate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from text_extraction import extract_text
from test_database import add_new_collection, get_documents
import chromadb

# use xichen2 env
load_dotenv()
model_path = "mistralai/mixtral-8x7b-instruct-v0.1"
collection = None
response = None
model = None
os.environ["REPLICATE_API_TOKEN"] = "r8_HXkl38P4AwS8wMIQeiM32odzsKNwrir3Ff4BH"

#Shreyas Code
# def initialise_model(model_path,device_map):
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"

#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map=device_map,
#         trust_remote_code=False,
#         cache_dir='./models',
#         revision="main"
#     )

#     scoring_pipeline = pipeline(
#         model=model,
#         tokenizer=tokenizer,
#         task="text-generation",
#         return_full_text=False,
#         temperature=0.1,
#         do_sample=True,
#         top_p=0.95,
#         top_k=40,
#         max_new_tokens=300
#     )
#     mistral_pipeline = HuggingFacePipeline(pipeline=scoring_pipeline)
#     return mistral_pipeline

def initialise_db(folder_path='./text_message_dataset'):
    chroma_client = chromadb.Client()
    collection = add_new_collection("ask_out_collection", folder_path, chroma_client)
    return collection

#run LLM on local machine to get response
async def generate_response(chat_history, model):
    prompt_template = "[INST]\n"+\
        "You are a dating advisor. You are given a chat history which is in list. The chat history is delimited by ###. In the list, there are sub-lists which contain a conversation chunk."+\
        "The first element of each sublist is user1. The other element is user 2.\n"+\
        "Give a grade on how interested user 1 is to user 2.\n"+\
        f"### {chat_history} \n"+\
        "Response:\n[/INST]"
    
    prompt = PromptTemplate.from_template(prompt_template)

    chain = prompt | model
    response = chain.invoke({"chat_history": chat_history})
    print(response)
    return response

# use replicate API to generate response
async def replicate_generation(chat_history):
    prompt_template = """
    You are a dating advisor. You will be given a chat history and you are required to give a overall score on how interested user 1 is to user 2.

    Your task is to evalaute based on 3 categories: capital letters used, the frequency of the text and the emojis used.

    The score for each of them should be between 0 to 10.

    Please output the score in the following format: "category: /10".
    
    The category should be one of the 3 categories. The score should be before the '/'.

    The same format should be followed for the frequency of the text and the emojis used.

    Please give a overall score based on the above three scores, which is between 0 to 30.
    -------------------------------------------
    Here is one example:

    User 1:

    Capital letters: 7/10
    Frequency: 8/10
    Emojis: 5/10
    User 2:

    Capital letters: 6/10
    Frequency: 8/10
    Emojis: 5/10

    Overall score for User 1: 20/30
    Overall score for User 2: 19/30
    -------------------------------------------
    The chat history is delimited by ###.

    ### {chat_history}

    Response:""".format(chat_history=chat_history)

    output = replicate.run(
        model_path,
        input={
            "top_k": 50,
            "top_p": 0.9,
            "prompt": prompt_template,
            "temperature": 0.3,
            "max_new_tokens": 1024,
            "prompt_template": "<s>[INST] {prompt} [/INST] ",
            "presence_penalty": 0,
            "frequency_penalty": 0
        }
    )
    output = "".join(output)
    return output

async def RAG(chat_history,collection):
    conversation_output = await replicate_generation(chat_history)
    #Query the collection
    query_text = ""
    results = get_documents(collection,query_text)

    prompt_template = """
    You are a dating advisor. You will be given a analysis of the conversation between two individuals.

    You are also given few references of conversation between two individuals who are using the same dating platform.

    You are required to use both the analysis of the conversation and the references to generate new responses for the conversation.

    The response should help the individuals have a higher chance of going on a date.
    
    The score for the conversation is delimited by ### and the references are delimited by &&&.
    ### {conversation_output}

    &&& {references}

    Response:""".format(conversation_output=conversation_output,references=results)

    output = replicate.run(
        model_path,
        input={
            "top_k": 50,
            "top_p": 0.9,
            "prompt": prompt_template,
            "temperature": 0.3,
            "max_new_tokens": 1024,
            "prompt_template": "<s>[INST] {prompt} [/INST] ",
            "presence_penalty": 0,
            "frequency_penalty": 0
        }
    )
    output = "".join(output)
    return output

def respond(message, chat_history):
    bot_message = random.choice(
        ["How are you?", "I love you", "I'm very hungry"])
    chat_history.append((message, bot_message))
    time.sleep(0.5)
    return "", chat_history


async def AI_interest_eval(chat_history):
    print("this is chat history", chat_history)
    stringed_chat = str(chat_history)
    print("string chat: ", stringed_chat)
    AI_response = await RAG(chat_history, collection)
    return AI_response


with gr.Blocks() as demo:
    with gr.Tab("Chatbot"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])
        AI_assessment_btn = gr.Button(value="AI interest evaluation")
        AI_review = gr.Textbox(label="AI's opinion on this conversation")


        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        AI_assessment_btn.click(
            AI_interest_eval, inputs=chatbot, outputs=AI_review)
        
    with gr.Tab("Image upload"):
        with gr.Row():
            image_input = gr.ImageEditor()
            image_output = gr.Textbox()
        image_button = gr.Button("Extract text from image")

        


    image_button.click(
        extract_text, inputs=image_input, outputs=image_output
    )

if __name__ == "__main__":
    collection = initialise_db()
    demo.launch(debug=True)
