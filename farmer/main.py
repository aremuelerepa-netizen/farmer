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

# MEMORY & SYSTEM PROMPT
# This stores the "AgroGuru" personality and the conversation context
chat_history = [
    {
        "role": "system", 
        "content": (
            "You are AgroGuru, a warm, friendly, and seasoned farming mentor. "
            "Your goal is to help farmers grow more while feeling supported. "
            "Guidelines:\n"
            "- PERSONALITY: Be encouraging. Use phrases like 'Let's look into that' or 'Keep at it, friend.'\n"
            "- STYLE: Simple, clear language. Use bullet points for steps.\n"
            "- BREVITY: Keep responses short and punchy. Max 80 words.\n"
            "- GURU TIP: Always end with a 'Guru Tip'—a small expert secret or traditional wisdom.\n"
            "- CONTEXT: It is January 2026. Mention seasonal tasks if relevant."
        )
    }
]

@app.get("/")
async def serve_interface():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(index_path) if os.path.exists(index_path) else {"error": "index.html not found"}

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    try:
        content = await file.read()
        base64_image = base64.b64encode(content).decode('utf-8')
        
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            messages=[
                {"role": "system", "content": "You are AgroGuru Pro. Identify Crop/Disease, Severity (1-5), and short Action steps. Be extremely concise. Max 100 words."},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Analyze this crop leaf."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        analysis = completion.choices[0].message.content
        # Add the diagnosis to memory so the bot remembers the plant you're talking about
        chat_history.append({"role": "assistant", "content": f"I analyzed a photo for you: {analysis}"})
        return {"analysis": analysis}
    except Exception as e:
        return {"error": "AI Vision busy. Please try again."}

@app.post("/chat")
async def chat_text(data: dict):
    try:
        user_text = data.get("message")
        
        # Add user message to history
        chat_history.append({"role": "user", "content": user_text})
        
        # Limit memory to save tokens (keep System Prompt + last 10 messages)
        context = [chat_history[0]] + chat_history[-10:]

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=context
        )
        
        reply = completion.choices[0].message.content
        # Save bot reply to history
        chat_history.append({"role": "assistant", "content": reply})
        
        return {"reply": reply}
    except Exception as e:
        return {"error": "Chat service busy."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
