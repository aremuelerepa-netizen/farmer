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

# --- THE AGROGURU PERSONALITY ---
# We force the AI to be warm and remember it is a mentor, not a robot.
SYSTEM_PROMPT = (
    "You are AgroGuru, a warm, wise, and very friendly farming mentor. "
    "Your tone is like a helpful neighbor sharing coffee. Use phrases like 'Bless your heart,' 'Let's see here,' or 'Don't you worry.' "
    "Rules: "
    "1. Be encouraging. 2. Use simple bullet points. 3. Max 80 words. "
    "4. If a scan was done recently, ALWAYS refer back to that specific plant. "
    "5. End with a 'Guru Tip' (traditional farming wisdom)."
)

# Global memory (Note: This clears if the server restarts on Render)
chat_memory = [{"role": "system", "content": SYSTEM_PROMPT}]

@app.get("/")
async def serve_interface():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(index_path) if os.path.exists(index_path) else {"error": "index.html not found"}

@app.post("/chat")
async def chat_text(data: dict):
    try:
        user_text = data.get("message")
        
        # Add user message to history
        chat_memory.append({"role": "user", "content": user_text})
        
        # We send the System Prompt + the whole history to ensure it stays 'Friendly'
        # and remembers the previous scan result.
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=chat_memory,
            temperature=0.8, # Higher temperature makes it more 'friendly' and less robotic
        )
        
        reply = completion.choices[0].message.content
        chat_memory.append({"role": "assistant", "content": reply})
        
        return {"reply": reply}
    except Exception as e:
        return {"error": "AgroGuru is taking a quick break. Try again!"}

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    try:
        content = await file.read()
        base64_image = base64.b64encode(content).decode('utf-8')
        
        # Vision Analysis
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            messages=[
                {"role": "system", "content": "You are a friendly crop doctor. Identify the plant and disease. Be warm. Max 90 words."},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Take a look at this plant for me, Guru."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        analysis_result = completion.choices[0].message.content

        # THE RE-INFORCEMENT: We add this as a System note so the AI CANNOT forget it.
        chat_memory.append({
            "role": "system", 
            "content": f"USER CONTEXT: The user just scanned a plant. You identified it as: {analysis_result}. You must remember this plant for all future questions until the chat is reset."
        })
        
        # Also add a friendly confirmation message from the Guru
        chat_memory.append({"role": "assistant", "content": analysis_result})
        
        return {"analysis": analysis_result}
    except Exception as e:
        return {"error": "Couldn't see that clearly. Try a smaller photo, friend."}

@app.post("/reset")
async def reset_chat():
    global chat_memory
    chat_memory = [{"role": "system", "content": SYSTEM_PROMPT}]
    return {"status": "Memory cleared"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
