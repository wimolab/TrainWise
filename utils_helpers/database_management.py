import faiss

def initialize_database(dimension:int) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(dimension)
    return index

def update_database(file_path:str, embeddings) -> faiss.IndexFlatIP:
    # Load database
    faiss_index = faiss.read_index(file_path)
    # Add new embeddings
    faiss_index.add(embeddings)
    # Save updated database
    faiss.write_index(faiss_index, file_path)


