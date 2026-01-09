import base64
import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq - Ensure the key is correct in Render Environment Variables
GROQ_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_KEY)

@app.get("/")
async def serve_interface():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(index_path) if os.path.exists(index_path) else {"error": "index.html not found"}

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    try:
        content = await file.read()
        base64_image = base64.b64encode(content).decode('utf-8')
        
        # Using the specific Llama 3.2 Vision model
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {"role": "system", "content": "You are a professional agronomist. Identify the plant and disease."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this crop image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
        )
        return {"analysis": completion.choices[0].message.content}
    except Exception as e:
        print(f"DIAGNOSE ERROR: {str(e)}") # This shows in Render Logs
        return {"error": "Vision AI is currently unavailable."}

@app.post("/chat")
async def chat_text(data: dict):
    try:
        user_text = data.get("message")
        # Using the most stable Llama 3 text model
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful AI farming assistant named AgroGuru."},
                {"role": "user", "content": user_text}
            ]
        )
        return {"reply": completion.choices[0].message.content}
    except Exception as e:
        print(f"CHAT ERROR: {str(e)}") # This shows in Render Logs
        return {"error": "The AI is having trouble connecting."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
