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

html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Cosine Similarity Calculator</title>
</head>
<body>
    <h1>Cosine Similarity Calculator</h1>
    <form method="post">
        <label for="text1">Text 1:</label>
        <input type="text" name="text1" id="text1" required><br>
        <label for="text2">Text 2:</label>
        <input type="text" name="text2" id="text2" required><br>
        <button type="submit">Calculate Cosine Similarity</button>
    </form>
    <br>
    <h2>Cosine Similarity:</h2>
    <p>{{ similarity }}</p>
</body>
</html>
"""

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
