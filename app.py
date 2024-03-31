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
# from langchain.chains import LLMChain
# from langchain_openai import ChatOpenAI

# use xichen2 env
load_dotenv()
model_path = "mistralai/mistral-7b-instruct-v0.2"
device_map = "auto"
response = None
# tokenizer = None
model = None
# model_path = "gpt-3.5-turbo-16k"
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#Shreyas Code
def initialise_model(model_path,device_map):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=False,
        cache_dir='./models',
        revision="main"
    )

    scoring_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        return_full_text=False,
        temperature=0.1,
        do_sample=True,
        top_p=0.95,
        top_k=40,
        max_new_tokens=300
    )
    mistral_pipeline = HuggingFacePipeline(pipeline=scoring_pipeline)
    return mistral_pipeline
    # llm_openai = ChatOpenAI(model_name=model_path, openai_api_key=OPENAI_API_KEY, temperature=0.3)
    # return llm_openai

#run LLM on local machine to get response
async def generate_response(chat_history, model):
    prompt_template = "[INST]\n"+\
        "You are a dating advisor. You are given a chat history which is in list. The chat history is delimited by ###. In the list, there are sub-lists which contain a conversation chunk."+\
        "The first element of each sublist is user1. The other element is user 2.\n"+\
        "Give a grade on how interested user 1 is to user 2.\n"+\
        f"### {chat_history} \n"+\
        "Response:\n[/INST]"
    
    prompt = PromptTemplate.from_template(prompt_template)

    # # ChatGPT model code
    # llm_chain = LLMChain(llm=model, prompt=prompt)

    chain = prompt | model
    response = chain.invoke({"chat_history": chat_history})
    print(response)
    return response

# use replicate API to generate response
async def replicate_generation(chat_history):
    prompt_template = "You are a dating advisor. You are given a chat history which is in list.\n"+\
        "The chat history is delimited by ###. In the list, there are sub-lists which contain a conversation chunk.\n"+\
        "The first element of each sublist is user1. The other element is user 2.\n"+\
        "Give a grade on how interested user 1 is to user 2.\n"+\
        f"### {chat_history} \n"+\
        "Response:\n"
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
    return output
#Xi Chen initial Code
# def initialise_model(model_path, device_map):

#     tokenizer = AutoTokenizer.from_pretrained(model_path)

#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map=device_map,
#         #  max_memory={ 0: "2GB", 1: "5GB", 2: "10GB"},
#         trust_remote_code=False,
#         revision="main"
#     )

#     return model, tokenizer



# async def generate_response(prompt, model, tokenizer):

#     prompt_template = f'''[INST] <<SYS>>" In the list, there are a set of tuples."\
#         "The first element of the tuple is user1. The other is user 2."\
#         "Give a grade on how interested user 1 is to user 2."\
#     <</SYS>>
#     {prompt}[/INST]
#     \n\n ### Response:'''

#     # General PROMPT template

#     # print("\n\n*** Generate:")
#     input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids
#     if torch.cuda.is_available():
#         input_ids = input_ids.to("cuda")

#     output = model.generate(inputs=input_ids, temperature=0.7,
#                             do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
#     response = tokenizer.decode(output[0])
#     print(response)

#     return response

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
    # AI_response = await generate_response(chat_history, model, tokenizer)
    # AI_response = await generate_response(chat_history, model)
    AI_response = await replicate_generation(chat_history)
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
            image_input = gr.Image()
            image_output = gr.Textbox()
        image_button = gr.Button("Extract text from image")

    image_button.click(
        extract_text, inputs=image_input, outputs=image_output)

if __name__ == "__main__":
    # model, tokenizer = initialise_model(model_path, device_map)
    model = initialise_model(model_path, device_map)
    demo.launch(debug=True)
