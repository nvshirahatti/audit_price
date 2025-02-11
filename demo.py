import gradio as gr
import json
import sqlite3
import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File
import uvicorn
import openai
from openai import AsyncOpenAI

# OpenAI API Key
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY
from openai import AsyncOpenAI
client = openai.Client(api_key=OPENAI_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Initialize SQLite database
conn = sqlite3.connect("audit_pricing.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS historical_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Address TEXT,
    BuildingArea REAL,
    PrimaryUseType TEXT,
    Jurisdiction TEXT,
    Size TEXT,
    Customer TEXT,
    AuditLevel INTEGER,
    AuditPrice REAL
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_data TEXT,
    predicted_price REAL,
    feedback TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# Store historical data globally
historical_data = None

@app.post("/upload_data")
def upload_historical_data(file: UploadFile = File(...)):
    global historical_data
    try:
        if isinstance(file, UploadFile):
            print("Received an UploadFile object.")
            content = file.file.read().decode("utf-8").strip()
        else:
            with open(file.name, 'r', encoding='utf-8') as f:
                print("Received a filepath.")
                content = f.read().strip()
        
        print("File Content:")
        print(content[:500])  # Print first 500 characters for debugging
        
        historical_data = pd.read_csv(io.StringIO(content), sep="\t", on_bad_lines="skip")
        print("Parsed Data:")
        print(historical_data.head())

        historical_data.to_sql("historical_data", conn, if_exists="replace", index=False)
        print("Data successfully stored in database.")
        
        historical_summary = historical_data.describe().to_json()
        print(historical_summary)
        insights_prompt = (
            f"Analyze the following historical audit pricing data and extract key insights, trends, and optimization opportunities: {historical_summary}. "
            "Summarize these insights in a structured JSON format for reinforcement learning (RL) and predictive pricing."
        )
        insights_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI trained in reinforcement learning and data analytics for audit pricing."},
                {"role": "user", "content": insights_prompt}
            ]
        )
        response_content = insights_response.choices[0].message.content.strip()
        extracted_json = {}
        if "```json" in response_content:
            try:
                extracted_json = json.loads(response_content.split("```json")[1].split("```")[0].strip())
            except json.JSONDecodeError:
                extracted_json = {"error": "Invalid JSON extracted from response"}
        insights_data = {"raw_text": response_content, "structured_data": extracted_json}
        
        print("hello world")
        print(insights_data)
        return {
            "status": "Historical data uploaded successfully", 
            "insights": insights_data
        }
    except Exception as e:
        error_message = {"error": f"Failed to process uploaded file: {str(e)}"}
        print("Returning Error:", error_message)
        return error_message

# Function to interact with OpenAI for audit pricing predictions
def get_dynamic_pricing(msg: str):
    global historical_data
    historical_summary = "No historical data available."
    
    if historical_data is not None:
        historical_summary = historical_data.describe().to_json()
    
    print(msg)
    try:
        prompt = (
            f"You are an RL-based audit pricing optimizer. "
            f"Based on these audit attributes: {msg}, "
            f"and historical audit pricing insights: {historical_summary}, "
            f"predict an optimal audit price and suggest dynamic pricing adjustments."
            f"You MUST return a JSON object called response with 'predicted_audit_price'. Respond back in json not text"
        )
        print("prompt")        
        print(prompt)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI trained to optimize audit pricing based on historical data and user feedback."},
                {"role": "user", "content": prompt}
            ]
        )
        print("response")        
        response_content = response.choices[0].message.content.strip()
        print(response_content)
        print("end response")
        return response_content
    except Exception as e:
        return {"predicted_audit_price": response.choices[0].message.content}  # Wrap plain text in JSON format

@app.post("/predict")
def predict_audit_price_api(msg: str):
    try:
        return {"predicted_audit_price": get_dynamic_pricing(msg)}
    except Exception as e:
        return {"error": f"Failed to generate prediction: {str(e)}"}

@app.post("/feedback")
def submit_feedback(request_data: str, predicted_price: float, feedback: str):
    cursor.execute(
        "INSERT INTO feedback (request_data, predicted_price, feedback) VALUES (?, ?, ?)",
        (request_data, predicted_price, feedback)
    )
    conn.commit()
    return {"status": "Feedback recorded successfully"}

def debug_upload(file):
    response = upload_historical_data(file)
    print("Upload Function Response:", response)  # Debugging Gradio input-output
    return response if isinstance(response, dict) else {"error": "Invalid response format"}

def handle_upload_response(response):
    print("Hello world")
    if isinstance(response, dict) and response.get("status") == "success":
        print("here")
        return response.get("message", "Upload successful")  # Extract message
    elif isinstance(response, dict) and response.get("status") == "error":
        return response.get("message", "Upload failed")  # Extract message
    else:
        return "Unexpected response from upload function" # Handle unexpected types


# Gradio UI
def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 🎯 AI-Powered RL Audit Pricing Model")
        chat_history = gr.State([])
        message_input = gr.Textbox(label="Enter JSON Data")
        chat_output = gr.Chatbot(label="Audit Pricing Chat")
        submit_button = gr.Button("Get Prediction")
        upload_button = gr.File(label="Upload Historical Data", type="filepath", file_types=[".tsv"])
        upload_submit_button = gr.Button("Upload Data")
        feedback_input = gr.Textbox(label="Provide Feedback on Prediction (Enter JSON Data)")
        feedback_submit_button = gr.Button("Submit Feedback")
        price_adjustment = gr.Slider(minimum=-1000, maximum=30000, step=50, label="Adjust Audit Price")
        
        submit_button.click(lambda msg, history, adjust: (
        (msg, str(predict_audit_price_api(json.loads(msg)).get("predicted_audit_price", "Error: Missing predicted_audit_price"))),
        history + [(msg, str(predict_audit_price_api(json.loads(msg)).get("predicted_audit_price", "Error: Missing predicted_audit_price")))]),
        inputs=[message_input, chat_history, price_adjustment], 
        outputs=[chat_output, chat_history])

        #upload_submit_button.click(lambda file: ("Upload Successful",) if isinstance(upload_historical_data(file), dict) else ("Upload Failed",), inputs=[upload_button], outputs=[chat_output])
        upload_submit_button.click(
            lambda file: (debug_upload(file),),  # Call handler
            inputs=[upload_button],
            outputs=[chat_output]
        )
        feedback_submit_button.click(submit_feedback, inputs=[message_input, feedback_input], outputs=[chat_output])
    return demo

# Run App
if __name__ == "__main__":
    import threading
    
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    gradio_app = build_interface()
    gradio_app.launch(share=True)
