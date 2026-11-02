from fastapi import FastAPI

app=FastAPI()

@app.get('/greet')
def greet(name="Guest"):
    return {f"Hello, {name}"}
