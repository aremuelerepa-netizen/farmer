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

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# THE SYSTEM PROMPT
AGRO_PROMPT = (
    "You are AgroGuru, a warm, seasoned farming mentor. "
    "Be encouraging and use simple language. "
    "Always use bullet points for steps. Max 80 words. "
    "CRITICAL: Always end your response with a 'Guru Tip'—a small expert secret."
)

# Global memory list
chat_memory = [{"role": "system", "content": AGRO_PROMPT}]

@app.get("/")
async def serve_interface():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(index_path) if os.path.exists(index_path) else {"error": "index.html not found"}

@app.post("/chat")
async def chat_text(data: dict):
    try:
        user_text = data.get("message")
        chat_memory.append({"role": "user", "content": user_text})
        
        # Send everything (history + prompt) to the AI
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=chat_memory[-12:] # Keep last 12 interactions for memory
        )
        
        reply = completion.choices[0].message.content
        chat_memory.append({"role": "assistant", "content": reply})
        return {"reply": reply}
    except Exception as e:
        return {"error": "Chat busy. Try again."}

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    try:
        content = await file.read()
        base64_image = base64.b64encode(content).decode('utf-8')
        
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            messages=[
                {"role": "system", "content": "Identify Crop/Disease & Action steps. Concise. Max 100 words."},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Analyze this crop leaf."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        res = completion.choices[0].message.content
        # Link the scan to the chat memory
        chat_memory.append({"role": "assistant", "content": f"USER UPLOADED PHOTO. ANALYSIS: {res}"})
        return {"analysis": res}
    except Exception as e:
        return {"error": "Vision error."}

@app.post("/reset")
async def reset_chat():
    global chat_memory
    chat_memory = [{"role": "system", "content": AGRO_PROMPT}]
    return {"status": "memory cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
