# Import the FastAPI library
from fastapi import FastAPI

# Create an instance of the FastAPI class
app = FastAPI()

# Define the root endpoint (GET request)
@app.get("/greet/{name}")
def greet(name: str):
    # This function will be called when someone accesses the URL with a name
    # It will return a JSON response with a personalized greeting message.
    return {"message": f"Hello, {name}!"}