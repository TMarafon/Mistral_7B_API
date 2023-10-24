from fastapi import FastAPI
from huggingface_hub import InferenceClient

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello world"}

@app.get("/{input}")
async def inference(input):
    client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.1")
    prompt = """<s>[INST] {0} [/INST]</s>""".format(input)
    res = client.text_generation(prompt, max_new_tokens=95)
    return res