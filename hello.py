# Import the FastAPI library
from fastapi import FastAPI

# Create an instance of the FastAPI class
app = FastAPI()

# Define the root endpoint (GET request) 
@app.get("/")
def read_root():
    # This function will be called when someone accesses the root ("/") URL.
    # It will return a JSON response with a message.
    return {"message": "Hello, World!"}