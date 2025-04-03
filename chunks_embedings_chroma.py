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
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI


# imports for langchain and Chroma and plotly
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma


# price is a factor for our company, so we're going to use a low cost model
MODEL = "gpt-4o-mini"
db_name = "vector_db"

# Load environment variables in a file called .env
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
openai = OpenAI()


# With massive thanks to student Dr John S. for fixing a bug in the below for Windows users!
context = {}
employees = glob.glob("knowledge-base/employees/*")
 
# Read in documents using LangChain's loaders
# Take everything in all the sub-folders of our knowledgebase

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
        
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

len(chunks)

doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"Document types found: {', '.join(doc_types)}")


# Put the chunks of data into a Vector Store that associates a Vector Embedding with each chunk
embeddings = OpenAIEmbeddings()


# Check if a Chroma Datastore already exists - if so, delete the collection to start from scratch
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

    
# Create our Chroma vectorstore!
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")


# Get one vector and find how many dimensions it has
collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")