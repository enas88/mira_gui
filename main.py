from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel

app = FastAPI()

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

class InputData(BaseModel):
    text1: str
    text2: str

class OutputData(BaseModel):
    similarity: float

templates = Jinja2Templates(directory="templates")



@app.get("/")
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/", response_model=OutputData)
async def calculate_similarity(input_data: InputData):
    # Encode text to get embeddings
    embeddings1 = model.encode(input_data.text1, convert_to_tensor=True)
    embeddings2 = model.encode(input_data.text2, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0]

    return {"similarity": similarity}
