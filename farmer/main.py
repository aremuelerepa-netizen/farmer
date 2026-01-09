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

# Initialize Groq with your Render Environment Variable
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/")
async def serve_interface():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(index_path) if os.path.exists(index_path) else {"error": "index.html not found"}

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    try:
        content = await file.read()
        base64_image = base64.b64encode(content).decode('utf-8')
        
        # NEW: Llama 4 Vision Model (2026 Stable Version)
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            messages=[
                {
                    "role": "system", 
                    "content": "You are a professional agronomist. Identify the plant and any disease in the image. Be concise."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is wrong with this plant?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]
        )
        return {"analysis": completion.choices[0].message.content}
    except Exception as e:
        print(f"Vision Error: {str(e)}")
        return {"error": "The Vision AI is currently undergoing scheduled maintenance. Please try again later."}

@app.post("/chat")
async def chat_text(data: dict):
    try:
        user_text = data.get("message")
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are AgroGuru, an expert farming assistant."},
                {"role": "user", "content": user_text}
            ]
        )
        return {"reply": completion.choices[0].message.content}
    except Exception as e:
        print(f"Chat Error: {str(e)}")
        return {"error": "Text service temporarily unavailable."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
