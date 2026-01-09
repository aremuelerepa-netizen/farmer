import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq
client = Groq(api_key="YOUR_GROQ_API_KEY")

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    try:
        content = await file.read()
        base64_image = base64.b64encode(content).decode('utf-8')

        # --- THE AGENTIC PROMPT ---
        # This tells the AI exactly how to behave
        system_instructions = (
            "You are an Elite AI Agronomist specializing in plant pathology. "
            "Analyze the image provided and respond with high precision. "
            "Your response must include:\n"
            "1. **Crop Identification**: Specify the plant species.\n"
            "2. **Diagnosis**: Identify the specific disease or pest issue.\n"
            "3. **Severity**: Estimate the damage (Low, Medium, High).\n"
            "4. **Action Plan**: Provide 3 clear steps for recovery (Organic and Chemical options).\n"
            "5. **Future Prevention**: One tip to avoid this in the next season.\n"
            "Keep the language professional but easy for a farmer to understand."
        )

        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": system_instructions
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please provide a detailed health report for this leaf."},
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            temperature=0.2,  # Low temperature keeps the AI factual and precise
            max_tokens=1024
        )

        return {"analysis": completion.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)