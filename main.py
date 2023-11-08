from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

persons = []

class Person(BaseModel):
    first_name: str
    last_name: str
    age: int

@app.get("/", response_class=HTMLResponse)
async def get_form():
    with open("templates/form.html") as f:
        html = f.read()
    return HTMLResponse(content=html)

@app.post("/add_person/")
async def add_person(person: Person):
    persons.append(person.dict())
    return {"message": "Person added successfully"}

@app.get("/list_persons/")
async def list_persons():
    return persons

@app.delete("/delete_person/{index}/")
async def delete_person(index: int):
    if 0 <= index < len(persons):
        deleted_person = persons.pop(index)
        return {"message": "Person deleted successfully", "deleted_person": deleted_person}
    else:
        return {"error": "Invalid index"}

@app.put("/update_person/{index}/")
async def update_person(index: int, person: Person):
    if 0 <= index < len(persons):
        persons[index] = person.dict()
        return {"message": "Person updated successfully"}
    else:
        return {"error": "Invalid index"}
