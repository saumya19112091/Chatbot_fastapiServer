from fastapi import FastAPI
from pydantic import BaseModel
from OpenAIAgent import OpenAIAgent
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import chat_memory_manager
from fastapi.responses import StreamingResponse
import os
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://chatbot-react-ui-new-ts.vercel.app"],  # Allows all origins
    allow_credentials= True,
    allow_methods=["*"],  # Allows all methods including POST
    allow_headers=["*"],  # Allows all headers
)

class userInput(BaseModel):
    user_input: str
    unique_session_id: str


@app.post('/openaiApi')
def Open_AI_Response_API(request: userInput):
    try:
        obj = OpenAIAgent()
        ai_response = obj.get_ai_response(request.user_input, request.unique_session_id)
        return StreamingResponse(ai_response, media_type="text/event-stream") 
    except Exception as e:
        return {"error": e}
    
@app.get('/')
def Home_screen_api():
    return "hello world"

@app.on_event('startup')
async def startup_event():
    print("startup api called")
    asyncio.create_task(chat_memory_manager.clean_up_memory())

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
