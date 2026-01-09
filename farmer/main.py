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

# API Key from Environment
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
        
        # 2026 STABLE VISION MODEL: Llama 4 Scout
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            messages=[
                {
                    "role": "system", 
                   "content": "You are AgroGuru Pro. Be extremely concise. 1. Identify Crop/Disease. 2. Severity (1-5). 3. Immediate Action (bullet points). 4. One Prevention tip. Max 100 words."
                    
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Analyze this crop leaf for diseases."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        return {"analysis": completion.choices[0].message.content}
    except Exception as e:
        print(f"Vision Error: {str(e)}")
        return {"error": "AI Vision model is currently unavailable."}

@app.post("/chat")
async def chat_text(data: dict):
    try:
        user_text = data.get("message")
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                   "content": (
    "You are AgroGuru, a warm, friendly, and seasoned farming mentor. "
    "Your goal is to help farmers grow more while feeling supported. "
    "Guidelines:\n"
    "- PERSONALITY: Be encouraging. Use phrases like 'Let's look into that,' 'Great question,' or 'Keep at it, friend.'\n"
    "- STYLE: Use simple, clear language. Avoid overly complex jargon unless you explain it.\n"
    "- STRUCTURE: Use bullet points for steps to make them easy to read on a phone in the field.\n"
    "- BREVITY: Keep responses short and punchy so the farmer can get back to work. Max 80 words.\n"
    "- GURU TIP: Always end with a 'Guru Tip'—a small, expert secret or a traditional wisdom tip (e.g., about companion planting or natural pest control).\n"
    "- CURRENT CONTEXT: It is January 2026. Mention seasonal tasks if relevant."
)
                }
                {"role": "user", "content": user_text}
            ]
        )
        return {"reply": completion.choices[0].message.content}
    except Exception as e:
        print(f"Chat Error: {str(e)}")
        return {"error": "Chat service busy."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)


