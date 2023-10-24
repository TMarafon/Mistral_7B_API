from fastapi import FastAPI
from huggingface_hub import InferenceClient
import os

app = FastAPI()

#@app.get("/")
#async def root():
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    token=os.environ.get("HF_TOKEN")
)

prompt = """[INST] What is your favourite condiment?  [/INST]"""

res = client.text_generation(prompt, max_new_tokens=95, details=True)
print(res)
#    return {"message": res}

@app.get("/{input}")
async def inference(input):
    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        token=os.environ.get("HF_TOKEN")
    )
    prompt = """[INST] {0} [/INST]""".format(input)
    print(prompt)
    res = client.text_generation(prompt, max_new_tokens=95)
    return res

