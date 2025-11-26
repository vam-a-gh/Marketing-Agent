from fastapi import FastAPI
from pydantic import BaseModel
from app.agent import marketing_graph


app = FastAPI(title="Marketing Agent API")

class Query(BaseModel):
    question: str

@app.post("/run-agent")
def run_agent(data: Query):
    output = marketing_graph.invoke({"question": data.question})
    return {"answer": output["final_answer"]}
