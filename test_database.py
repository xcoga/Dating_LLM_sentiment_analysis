import chromadb


chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")

collection.add(
    documents=["I like apples", "I fucking love watermelons"],
    metadatas=[{"source": "Xichen"}, {"source": "xcoga"}],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["Who like apples?"],
    n_results=1
)

print(results)
