from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello word"}

@app.get("{input}")
async def inference(input):
    return "Your input: " + input