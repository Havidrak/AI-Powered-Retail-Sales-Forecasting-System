import os
import secrets
from pathlib import Path
from uuid import uuid4
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from werkzeug.utils import secure_filename

# --- AI IMPORTS ---


# --- IMPORT YOUR LOGIC MODULES ---
from logic_business import analyze_sales_data 
from logic_customer import generate_dashboard_data 
from logic_stocks import process_restock_data

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') or secrets.token_hex(32)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = str(Path(__file__).resolve().parent / 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==========================================
# 🧠 AI MODEL SETUP (LLAMA 3.1 / TINYLLAMA)
# ==========================================
HF_TOKEN = os.getenv("HF_TOKEN")
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
chatbot_pipeline = None
_chatbot_attempted = False

def get_chatbot():
    global chatbot_pipeline, _chatbot_attempted
    if os.getenv("ENABLE_CHATBOT", "0") != "1":
        return None
    if not _chatbot_attempted:
        _chatbot_attempted = True
        try:
            import torch
            from transformers import pipeline
            chatbot_pipeline = pipeline("text-generation", model=model_id,
                model_kwargs={"torch_dtype": torch.float32}, device=-1, token=HF_TOKEN)
        except Exception:
            app.logger.exception("Chatbot initialization failed")
    return chatbot_pipeline

# ==========================================
# 🌐 ROUTES
# ==========================================

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            if not filename or Path(filename).suffix.lower() not in {".csv", ".xlsx"}:
                flash("Please upload a CSV or XLSX file.")
                return redirect(request.url)
            filename = f"{uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            session['current_file'] = filename
            return redirect(url_for('business_dashboard'))
            
    return render_template('upload.html')

@app.route('/dashboard/business')
def business_dashboard():
    filename = session.get('current_file')
    if not filename: return redirect(url_for('upload'))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    analysis = analyze_sales_data(filepath) 
    if not analysis: return redirect(url_for('upload'))
    return render_template('business.html', analysis_json=json.dumps(analysis))

@app.route('/dashboard/customer')
def customer_dashboard():
    filename = session.get('current_file')
    if not filename: return redirect(url_for('upload'))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result = generate_dashboard_data(filepath)
    if "error" in result: return redirect(url_for('upload'))
    return render_template('customer.html', data=result)

@app.route('/dashboard/stocks')
def stocks_dashboard():
    filename = session.get('current_file')
    if not filename: return redirect(url_for('upload'))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    report, error = process_restock_data(filepath, default_stock=100)
    if error: return redirect(url_for('upload'))
    return render_template('stocks.html', data=report)

# ==========================================
# 💬 FINAL CHATBOT (Synced with Dashboard + Multi-Tab Support)
# ==========================================
@app.route('/ask', methods=['POST'])
def chat_ask():
    filename = session.get('current_file')
    if not filename:
        return jsonify({"answer": "Please upload a file first."})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("question"), str) or not payload["question"].strip():
        return jsonify({"answer": "Please enter a question."}), 400
    user_q = payload["question"].strip()[:2000]
    
    facts = []

    # --- SOURCE 1: ASK THE DASHBOARD LOGIC (The Truth) ---
    try:
        # 1. Get Business Data (Revenue, Best Sellers)
        biz = analyze_sales_data(filepath)
        if biz and 'metrics' in biz:
            if 'total_revenue' in biz['metrics']:
                facts.append(f"Official Total Revenue: ${biz['metrics']['total_revenue']:,.2f}")
        if biz and 'ai_summary' in biz:
            if 'top_category' in biz['ai_summary']:
                facts.append(f"Official Best Selling Category: {biz['ai_summary']['top_category']}")
            if 'trend' in biz['ai_summary']:
                facts.append(f"Official Sales Trend: {biz['ai_summary']['trend']}")

        # 2. Get Customer Data (This fixes your 369 vs 32 issue!)
        cust = generate_dashboard_data(filepath)
        if cust and 'summary' in cust:
            facts.append(f"Official Total Customers: {cust['summary']['total_customers']}")
            facts.append(f"Official Loyal Members: {cust['summary']['total_members']}")

    except Exception as e:
        print(f"Logic Error: {e}")

    # --- SOURCE 2: READ RAW EXCEL (ALL TABS) ---
    # Only runs if we need deep details not in the dashboard
    try:
        if filepath.endswith('.csv'):
            dfs = {'Sheet1': pd.read_csv(filepath)}
        else:
            dfs = pd.read_excel(filepath, sheet_name=None) # Reads ALL sheets

        # Find the sheet with the most data (The real transaction sheet)
        real_df = None
        max_rows = 0
        for name, df in dfs.items():
            if len(df) > max_rows:
                max_rows = len(df)
                real_df = df
        
        if real_df is not None:
            # Clean columns
            real_df.columns = [str(c).lower().strip() for c in real_df.columns]
            
            # Find the most frequent product (Raw Calculation)
            prod_col = next((c for c in real_df.columns if c in ['product', 'item', 'sku', 'description']), None)
            if prod_col:
                best_item = real_df[prod_col].mode()[0]
                facts.append(f"Most Frequent Item (Raw Data): {best_item}")
            
            # Add row count as a backup fact
            facts.append(f"Total Rows in File: {max_rows}")

    except Exception as e:
        facts.append("Could not read raw file.")

    # --- 3. GENERATE ANSWER ---
    facts_text = "\n".join(facts)
    
    prompt = f"""<|system|>
You are a helpful Retail Assistant.
Use these OFFICIAL FACTS to answer the user.
FACTS:
{facts_text}

User Question: {user_q}

RULES:
1. Trust the "Official" numbers above over anything else.
2. Answer directly in one sentence.
3. Do not write code.
</s>
<|assistant|>"""

    chatbot = get_chatbot()
    if chatbot:
        try:
            outputs = chatbot(
                prompt, 
                max_new_tokens=150,
                do_sample=True,
                temperature=0.1, 
                repetition_penalty=1.2
            )
            full_text = outputs[0]["generated_text"]
            if "<|assistant|>" in full_text:
                ai_response = full_text.split("<|assistant|>")[-1].strip()
            else:
                ai_response = full_text.replace(prompt, "").strip()
                
            # Anti-Hallucination Filter
            if "```" in ai_response or "def " in ai_response:
                ai_response = f"According to the dashboard, {facts[0] if facts else 'the data is ready'}."

        except Exception as e:
            ai_response = "Sorry, I am having trouble speaking."
    else:
        ai_response = f"AI model disabled or unavailable. Dashboard facts: \n{facts_text}"

    return jsonify({"answer": ai_response})

if __name__ == '__main__':
    app.run(debug=False, host="127.0.0.1", port=5000)