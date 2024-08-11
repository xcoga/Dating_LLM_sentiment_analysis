from text_extraction.text_extraction import extract_text
from modules.AI_analyser import AI_interest_eval
from flask import Flask, request, jsonify
import os
from PIL import Image
import tempfile
from modules.db_interactions import initialise_db

app = Flask(__name__)
collection = initialise_db()


@app.route("/")
def hello_world():
    return jsonify({"Hello": "Hello world!"}, 200)


@app.route('/upload', methods=['POST'])
def upload_image():
    print("received POST request")

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        result = process_image(file)
        return jsonify({"result": result}), 200


def process_image(file):

    # YOLO cannot read Pillow objects. Hence, we save it as a temp jpeg file for YOLO to read.
    image = Image.open(file)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        image.save(temp_file.name)
        temp_file_path = temp_file.name

    chat = extract_text(temp_file_path)
    AI_review = AI_interest_eval(chat, collection)

    os.remove(temp_file_path)

    return AI_review


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
