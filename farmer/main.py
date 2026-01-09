import base64
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import uvicorn

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connects to Groq using the Environment Variable
api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

# ROUTE 1: Serve the Interface
@app.get("/", response_class=HTMLResponse)
async def serve_interface():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found. Please ensure it is in the same folder as main.py</h1>"

# ROUTE 2: Vision Analysis
@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    try:
        content = await file.read()
        base64_image = base64.b64encode(content).decode('utf-8')

        system_instructions = (
            "You are an Elite AI Agronomist. Identify the plant and disease. "
            "Provide a 3-step action plan including organic and chemical solutions."
        )

        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {"role": "system", "content": system_instructions},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this leaf photo for disease diagnosis."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.2
        )
        return {"analysis": completion.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Render uses port 10000 by default
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
