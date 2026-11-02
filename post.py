# Import the FastAPI library
from fastapi import FastAPI
from pydantic import BaseModel

# Create an instance of the FastAPI class
app = FastAPI()

# Define a Pydantic model for the user data (name and age)
class User(BaseModel):
    name: str
    age: int

# Define a POST endpoint to create a user
@app.post("/create_user/")
def create_user(user: User):
    # Return a message with the user's name and age
    return {"message": f"User {user.name} created, age {user.age}"}