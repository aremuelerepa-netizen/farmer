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
                    "content": (
                        "You are the AgroGuru Pro, a senior plant pathologist and agronomist. "
                        "When you see a plant image:\n"
                        "1. IDENTIFY: State the exact crop variety and the likely disease/pest.\n"
                        "2. SEVERITY: Rate the infection from Level 1 (Minor) to Level 5 (Critical).\n"
                        "3. CAUSE: Briefly explain why this happened (e.g., high humidity, soil deficiency).\n"
                        "4. ACTION PLAN: Provide 3 immediate organic steps and 1 chemical backup if necessary.\n"
                        "5. PREVENTION: One tip to stop this from returning next season.\n"
                        "Use a professional, urgent, yet encouraging tone. Be precise—don't say 'it might be,' say 'the symptoms suggest...'"
                    )
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
                        "You are AgroGuru Pro, a world-class farming consultant. "
                        "You specialize in high-yield, sustainable farming. "
                        "When a user asks a question:\n"
                        "- Give advice specific to the current season (January 2026).\n"
                        "- If they ask about planting, mention soil pH and spacing.\n"
                        "- If they ask about profit, mention market trends.\n"
                        "- Always include a 'Guru Tip'—a small, expert secret that most farmers miss.\n"
                        "- Keep responses concise and use bullet points for readability."
                    )
                },
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
