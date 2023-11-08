from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

persons = []

class Person(BaseModel):
    first_name: str
    last_name: str
    age: int

@app.get("/")
async def get_form():
    return FileResponse("templates/form.html")

@app.post("/person/")
async def add_person(person: Person):
    persons.append(person.model_dump())
    return {"message": "Person added successfully"}

@app.get("/person/")
async def list_persons():
    return persons

@app.get("/person/{index}/")
async def get_person(index: int):
    return persons[index]

@app.delete("/person/{index}/")
async def delete_person(index: int):
    if 0 <= index < len(persons):
        deleted_person = persons.pop(index)
        return {"message": "Person deleted successfully", "deleted_person": deleted_person}
    else:
        return {"error": "Invalid index"}

@app.put("/person/{index}/")
async def update_person(index: int, person: Person):
    if 0 <= index < len(persons):
        persons[index] = person.model_dump()
        return {"message": "Person updated successfully"}
    else:
        return {"error": "Invalid index"}
    

# Directory to store uploaded CSV files
UPLOAD_DIR = "uploads"

# A list to store CSV records
csv_records = []
[csv_records.append({"name": filename}) for filename in os.listdir(UPLOAD_DIR)]

# Route to list CSV records
@app.get("/csv/")
async def list_csv_records():
    return csv_records

# Route to upload a CSV file
@app.post("/csv/")
async def upload_csv_file(file: UploadFile):
    # Ensure the uploads directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    csv_records.append({"name": file.filename})
    return JSONResponse(content={"message": "CSV file uploaded successfully"}, status_code=201)