#TODO to be updated once we implement the db
import chromadb
import os

# Define folder path
def initialise_db(folder_path='./text_message_dataset'):
    chroma_client = chromadb.Client()
    collection = add_new_collection("ask_out_collection", folder_path, chroma_client)
    return collection

# Function to read text files from a folder
def read_text_files(folder_path):
    file_contents = []
    file_names = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8', errors='ignore') as file:
                # print(file_name)
                file_contents.append(file.read())
                file_names.append(file_name)
    return file_contents, file_names


def add_new_collection(collection_name, folder_path, client):
    # Read text files
    file_contents, file_names = read_text_files(folder_path)
    print(file_names)
    # print(file_contents)

    # Initialize Chroma client and create collection
    collection = client.create_collection(name=collection_name)

    # Add documents to Chroma collection with file names as metadata
    collection.add(
        documents=file_contents,
        metadatas=[{"source": file_name} for file_name in file_names],
        ids=[f"id{i}" for i in range(1, len(file_names) + 1)]
    )
    return collection

def get_documents(collection,query):
    #Query the collection
    results = collection.query(
        query_texts=[query],
        n_results=2
    )
    return results['documents']



