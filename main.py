from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
    