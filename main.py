from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from huggingface_hub import InferenceClient

from langchain.document_loaders import OnlinePDFLoader

import os

def prepareVectorDatabase():
    loader = OnlinePDFLoader(
        'https://huggingface.co/spaces/tmarafon2/Mistral_7b_API/resolve/main/files/Linkedin.pdf'    
    )
    docs = loader.load()
    print(docs[0].metadata)
    return docs[0].page_content[:200]


prepareVectorDatabase()

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
    prompt = """[INST] Tell me a story about space conquer. [/INST]""".format(input)
    return StreamingResponse(streamInference(prompt), media_type="text/plain")

@app.get("/{input}")
async def inference(input):
    prompt = """[INST] {0} [/INST]""".format(input)
    return StreamingResponse(streamInference(prompt), media_type="text/plain")

