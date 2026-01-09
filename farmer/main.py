import base64
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import uvicorn

app = FastAPI()

# 1. Enable CORS (Essential for mobile/browser security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Connect to Groq API
# Make sure "GROQ_API_KEY" is set in Render Environment Variables
api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

# 3. ROUTE: Serve the Main Interface
# This is what shows the chat UI when you visit the URL
@app.get("/")
async def serve_interface():
    # This looks for index.html in the same folder as main.py
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return {"error": "index.html file not found in root directory"}

# 4. ROUTE: AI Diagnosis (Agent Logic)
@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    try:
        # Check if API Key exists
        if not api_key:
            return {"analysis": "Error: API Key is missing. Add it to Render Environment Variables."}

        content = await file.read()
        base64_image = base64.b64encode(content).decode('utf-8')

        # Precise Expert Prompt
        system_instructions = (
            "You are an Elite AI Agronomist. Identify the plant and the disease. "
            "Provide a 3-step action plan including organic and chemical solutions. "
            "Keep the tone helpful and professional."
        )

        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {"role": "system", "content": system_instructions},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Diagnose this plant health issue."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.2
        )
        return {"analysis": completion.choices[0].message.content}
    except Exception as e:
        print(f"Server Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Render sets the PORT variable automatically
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
