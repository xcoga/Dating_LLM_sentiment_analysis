import gradio as gr
from text_extraction.text_extraction import extract_text
from modules.AI_analyser import AI_interest_eval


with gr.Blocks() as demo:
        
    with gr.Tab("Image upload"):
        with gr.Row():
            image_input = gr.File(file_count = 'single', label = "upload your jpeg file here")
            image_output = gr.Textbox()
        image_button = gr.Button("Extract text from image")
        AI_assessment_btn = gr.Button(value="AI interest evaluation")
        AI_review = gr.Textbox(label="AI's opinion on this conversation")

    image_button.click(
        extract_text, inputs=image_input, outputs=image_output
    )
    AI_assessment_btn.click(
            AI_interest_eval, inputs=image_output, outputs=AI_review)
if __name__ == "__main__":
    demo.launch(debug=True)
