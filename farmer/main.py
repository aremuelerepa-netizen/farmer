import base64
import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI()

# Enable CORS for mobile/web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- AGROGURU CONFIGURATION ---
SYSTEM_PROMPT = (
    "You are AgroGuru, a warm, seasoned farming mentor. "
    "Be encouraging and use simple language. "
    "Always use bullet points for steps. Max 80 words. "
    "CRITICAL: Always end your response with a 'Guru Tip'—a small expert secret or traditional wisdom."
)

# This list holds the conversation. 
# It starts with the System Prompt so the AI knows its personality.
chat_memory = [{"role": "system", "content": SYSTEM_PROMPT}]

@app.get("/")
async def serve_interface():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(index_path) if os.path.exists(index_path) else {"error": "index.html not found"}

@app.post("/chat")
async def chat_text(data: dict):
    try:
        user_text = data.get("message")
        
        # 1. Add user message to memory
        chat_memory.append({"role": "user", "content": user_text})
        
        # 2. Send the entire memory (including system prompt and previous scans)
        # We send the last 15 messages to ensure it doesn't forget the crop
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=chat_memory[-15:] 
        )
        
        reply = completion.choices[0].message.content
        
        # 3. Add AI reply to memory
        chat_memory.append({"role": "assistant", "content": reply})
        
        return {"reply": reply}
    except Exception as e:
        print(f"Chat Error: {e}")
        return {"error": "AgroGuru is resting. Try again in a moment."}

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    try:
        content = await file.read()
        base64_image = base64.b64encode(content).decode('utf-8')
        
        # 1. Vision Analysis
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            messages=[
                {"role": "system", "content": "You are a crop pathologist. Identify the plant and disease. Give short action steps. Max 100 words."},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "What is wrong with this plant?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        analysis_result = completion.choices[0].message.content

        # 2. KEY STEP: Inject the diagnosis into Chat Memory
        # We tell the 'Chat brain' exactly what the 'Vision brain' saw.
        chat_memory.append({
            "role": "assistant", 
            "content": f"SCAN REPORT: The user uploaded a photo. I identified it as: {analysis_result}. I will remember this for future questions."
        })
        
        return {"analysis": analysis_result}
    except Exception as e:
        print(f"Vision Error: {e}")
        return {"error": "Could not see the photo clearly. Try a smaller file."}

@app.post("/reset")
async def reset_chat():
    global chat_memory
    # Clear memory but put the System Prompt back in
    chat_memory = [{"role": "system", "content": SYSTEM_PROMPT}]
    return {"status": "Memory cleared"}

if __name__ == "__main__":
    # Get port from environment (required for Render/Heroku)
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
