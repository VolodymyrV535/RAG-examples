# from LLM course on Udemy
# Expert Knowledge Worker
# A question answering agent that is an expert knowledge worker
# To be used by employees of Insurellm, an Insurance Tech company
# The agent needs to be accurate and the solution should be low cost.
# This project will use RAG (Retrieval Augmented Generation) to ensure our question/answering assistant has high accuracy.
# This first implementation will use a simple, brute-force type of RAG..

# imports
import os
import glob

# imports for langchain
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

# price is a factor for our company, so we're going to use a low cost model
db_name = "vector_db"

## Read in documents using LangChain's loaders
# Take everything in all the sub-folders of our knowledgebase
# Thank you Mark D. and Zoya H. for fixing a bug here..

folders = glob.glob("knowledge-base/*")

# With thanks to CG and Jon R, students on the course, for this fix needed for some users 
text_loader_kwargs = {'encoding': 'utf-8'}
# If that doesn't work, some Windows users might need to uncomment the next line instead
# text_loader_kwargs={'autodetect_encoding': True}

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)
        
print(len(documents))
print(documents[24])

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(len(chunks))
print(chunks[10])

doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"Document types found: {', '.join(doc_types)}")

for chunk in chunks:
    if 'CEO' in chunk.page_content:
        print(chunk)
        print("_________")