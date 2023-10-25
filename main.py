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


import os

def prepareVectorDatabase():
    if 'qa' in globals():
        return
    
    loader = PyPDFDirectoryLoader('files')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
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
    return "Ready!" #qa({"query": "Who is Thiago Marafon?"})

prompts = [
    "Summarize Thiago's career in a few words",
    "Describe in bullet points Thiago's education",
    "Describe in bullet points Thiago's work experiences",
    "Describe in bullet points Thiago's certifications",
    "Describe in Thiago's role at Youper Inc",
    "Describe in Thiago's role at Softplan",
    "Describe in Thiago's coding skills",
    "Describe in Thiago's management skills",
]

async def streamInference():
    for p in prompts:
        result = qa({"query": p})
        yield result["query"]
        yield "\n"
        yield result["result"]
        yield "\n\n"


app = FastAPI()

@app.get("/")
async def root():
    prepareVectorDatabase()
    return StreamingResponse(streamInference(), media_type="text/plain")


@app.get("/{input}")
async def inference(input):
    #prompt = """[INST] {0} [/INST]""".format(input)
    #return StreamingResponse(streamInference(prompt), media_type="text/plain")
    if 'qa' not in globals():
        prepareVectorDatabase()
    return qa({"query": input})
