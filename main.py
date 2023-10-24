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

prompt = """<s>[INST] What is your favourite condiment?  [/INST]</s>"""

res = client.text_generation(prompt, max_new_tokens=95)
print(res)
#    return {"message": res}

@app.get("/{input}")
async def inference(input):
    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        token=os.environ.get("HF_TOKEN")
    )
    prompt = """<s>[INST] {0} [/INST]</s>""".format(input)
    print(prompt)
    res = client.text_generation(prompt, max_new_tokens=95)
    return res

