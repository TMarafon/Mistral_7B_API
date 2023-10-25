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

from langchain.chains import RetrievalQA


import os

def prepareVectorDatabase():
    loader = PyPDFDirectoryLoader('files')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceHubEmbeddings()
    db = Chroma.from_documents(docs, embeddings)
    
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1", 
        model_kwargs={"temperature":0.1, "max_new_tokens":300}
    )
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=db.as_retriever(
            search_type = "mmr", search_kwargs={
                "k":3,
                "score_threshold": .5
            })
        )
    global qa 
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=compression_retriever, 
        return_source_documents=True
    )
    return qa({"query": "Who is Thiago Marafon?"})


async def streamInference(prompt):
    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        token=os.environ.get("HF_TOKEN")
    )
    
    res = client.text_generation(prompt, max_new_tokens=200, stream=True, return_full_text=False)
    for r in res: 
      yield r

app = FastAPI()

@app.get("/")
async def example():
    return prepareVectorDatabase()

@app.get("/{input}")
async def inference(input):
    #prompt = """[INST] {0} [/INST]""".format(input)
    #return StreamingResponse(streamInference(prompt), media_type="text/plain")
    return qa({"query": input})
