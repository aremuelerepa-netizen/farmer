import base64
import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI()

# Enable CORS for mobile and browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq with your API Key
# IMPORTANT: Ensure GROQ_API_KEY is set in your Render Environment Variables
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 1. SERVE THE INTERFACE (Shows the index.html)
@app.get("/")
async def serve_interface():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "index.html not found in root directory"}

# 2. VISION ANALYSIS (For Photos)
@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    try:
        content = await file.read()
        base64_image = base64.b64encode(content).decode('utf-8')
        
        # FIXED: Using llama-3.2-11b-vision-preview (the active model)
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a professional AI Agronomist. Identify the plant and disease from the image and provide a 3-step action plan."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please analyze this leaf for diseases."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            temperature=0.2
        )
        return {"analysis": completion.choices[0].message.content}
    except Exception as e:
        print(f"Vision Error: {str(e)}")
        return {"error": "AI Vision is updating. Please try again in 30 seconds."}

# 3. TEXT CHAT (For Messages)
@app.post("/chat")
async def chat_text(data: dict):
    try:
        user_text = data.get("message")
        # FIXED: Using llama-3.3-70b-versatile (the active model)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are AgroGuru, a helpful and expert farming assistant."},
                {"role": "user", "content": user_text}
            ]
        )
        return {"reply": completion.choices[0].message.content}
    except Exception as e:
        print(f"Chat Error: {str(e)}")
        return {"error": "The AI is currently busy. Try again shortly."}

if __name__ == "__main__":
    # Render provides the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
