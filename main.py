from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from huggingface_hub import InferenceClient

from langchain.llms import HuggingFaceHub

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain.chains import RetrievalQA

def prepare_vector_database():
    if 'qa' in globals():
        return
    
    loader = PyPDFDirectoryLoader('files')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceHubEmbeddings()
    db = Chroma.from_documents(docs, embeddings)
    
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1", 
        model_kwargs={"temperature":0.1, "max_new_tokens":300}
    )
    #compressor = LLMChainExtractor.from_llm(llm)
    #compression_retriever = ContextualCompressionRetriever(
    #    base_compressor=compressor,
    #    base_retriever=db.as_retriever(
    #        search_type = "mmr", search_kwargs={
    #            "k":3,
    #            "score_threshold": .5
    #        })
    #    )

    #retriever_from_llm = MultiQueryRetriever.from_llm(
    #    retriever=db.as_retriever(), llm=llm
    #)

    retriever = db.as_retriever(
        search_type = "mmr", search_kwargs={
            "k":4,
            "score_threshold": .95
        }
    )
    global qa 
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    print('VectorDB ready!')
    return "Ready!"

prompts = [
    "Summarize Thiago's career in a 250 words",
    "Tell me in bullet points Thiago's education, citing only the intitution names, degree, and period",
    "Tell me in bullet points Thiago's work experiences, citing only the companies' names, title and period",
    "Tell me about Thiago's role at Youper Inc",
    "Tell me about Thiago's role at Softplan",
    "Tell me about Thiago's coding skills",
    "Tell me about Thiago's management skills",
]

async def stream_inference():
    for p in prompts:
        result = qa({"query": p})
        yield result["query"]
        yield "\n"
        yield result["result"]
        yield "\n\n"


app = FastAPI()

@app.get("/")
async def root():
    prepare_vector_database()
    return StreamingResponse(stream_inference(), media_type="text/plain")


@app.get("/{input}")
async def inference(input):
    prepare_vector_database()
    return qa({"query": input})
