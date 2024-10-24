import os
import re
import io
import logging
import json
from datetime import datetime

from flask import Flask, render_template, request, send_file, redirect, url_for
import yt_dlp
import markdown
import requests
import openai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

load_dotenv()
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
MOONSHOT_API_URL = "https://api.moonshot.cn/v1/"  # Replace with the actual KIMI API endpoint

client = openai.OpenAI(api_key=MOONSHOT_API_KEY, base_url=MOONSHOT_API_URL)
models = ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]

def choose_model(messages):
    logging.info("Estimating token count for model selection")
    # Use the Moonshot API to estimate token count
    response = requests.post(
        'https://api.moonshot.cn/v1/tokenizers/estimate-token-count',
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MOONSHOT_API_KEY}"
        },
        json={
            "model": "moonshot-v1-8k",
            "messages": messages
        }
    )
    
    if response.status_code == 200:
        token_count = response.json().get('data', {}).get('total_tokens', 0)
        logging.info(f"Token count from API: {token_count}")
    else:
        # Fallback to a simple estimation if API call fails
        token_count = len("".join([message["content"] for message in messages]).split()) * 1.3  # Rough estimate
        logging.warning(f"Failed to get token count from API. Using fallback estimation: {token_count}")
    
    if token_count <= 8000:
        model = models[0]  # moonshot-v1-8k
    elif token_count <= 32000:
        model = models[1]  # moonshot-v1-32k
    else:
        model = models[2]  # moonshot-v1-128k
    
    logging.info(f"Selected model: {model}")
    return model

def download_transcript(youtube_url):
    logging.info(f"Downloading transcript for URL: {youtube_url}")
    ydl_opts = {
        'writeautomaticsub': True,
        'skip_download': True,
        'outtmpl': 'transcript',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            video_title = info.get('title', 'Unknown Title')
        
        # Read the transcript
        with open('transcript.en.vtt', 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        logging.info("Transcript downloaded successfully")
        
        # Clean the transcript
        cleaned_transcript = clean_transcript(transcript)
        
        with open('cleaned_transcript.txt', 'w', encoding='utf-8') as f:
            f.write(cleaned_transcript)
        
        logging.info("Transcript cleaned and saved")
        return cleaned_transcript, video_title
    except Exception as e:
        logging.error(f"Error downloading transcript: {str(e)}")
        raise

def clean_transcript(transcript):
    logging.info("Cleaning transcript")
    # Remove header
    transcript = re.sub(r'^WEBVTT\n\n', '', transcript)
    
    # Remove timestamps and other non-relevant information
    cleaned_transcript = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*\n', '', transcript)
    cleaned_transcript = re.sub(r'<[^>]+>', '', cleaned_transcript)
    
    # Remove alignment and position information
    cleaned_transcript = re.sub(r'align:start position:0%\n', '', cleaned_transcript)
    
    # Remove extra newlines and leading/trailing whitespace
    cleaned_transcript = re.sub(r'\n{3,}', '\n\n', cleaned_transcript)
    cleaned_transcript = cleaned_transcript.strip()
    
    # Join lines that were split due to caption formatting
    lines = cleaned_transcript.split('\n')
    
    unique_lines = []
    i = 0
    while i < len(lines)-1:
        unique_lines.append(lines[i])
        for j in range(1,3):
            if not lines[i+j].startswith(lines[i]):
                break
        i += j
    
    cleaned_transcript = '\n'.join(unique_lines)
    # Remove multiple consecutive newlines
    cleaned_transcript = re.sub(r'\n{2,}', '\n', cleaned_transcript)
    
    logging.info("Transcript cleaning completed")
    return cleaned_transcript

def check_balance():
    logging.info("Checking balance")
    response = requests.get(
        'https://api.moonshot.cn/v1/users/me/balance',
        headers={
            "Authorization": f"Bearer {MOONSHOT_API_KEY}"
        }
    )
    
    if response.status_code == 200:
        balance_data = response.json().get('data', {})
        available_balance = balance_data.get('available_balance', 0)
        voucher_balance = balance_data.get('voucher_balance', 0)
        cash_balance = balance_data.get('cash_balance', 0)
        logging.info(f"Balance retrieved: Available: {available_balance}, Voucher: {voucher_balance}, Cash: {cash_balance}")
    else:
        logging.error(f"Failed to retrieve balance. Status code: {response.status_code}")
        available_balance = voucher_balance = cash_balance = 0
    
    return {
        "available_balance": available_balance,
        "voucher_balance": voucher_balance,
        "cash_balance": cash_balance
    }

def generate_summary(transcript, reading_time, video_title):
    logging.info(f"Generating summary for {reading_time} minute read")
    prompt = f"将下面Podcast的内容总结成 {reading_time} 分钟 的阅读材料，使用Markdown语法来增强材料的可读性。\n"
    prompt += "内容总结的语言为简体中文，当遇到英文人名时保留英文，遇到专业术语时，使用中文(English)的方式同时保留翻译和原文。\n"
    prompt += f"这个Podcast的标题是{video_title}，在整理这份内容总结时，首先总结为何podcast的题目是这个，然后再进行整个内容的总结。\n"
    prompt += "Podcast 内容如下：\n\n"
    prompt += transcript
    messages = [
        {"role": "system", "content": "请你扮演一个有丰富经验的研究人员。"},
        {"role": "user", "content": prompt},
    ]
    
    model = choose_model(messages)
    logging.info(f"Using model: {model}")
    
    with open('promt.txt', 'w', encoding='utf-8') as f:
        f.write(prompt)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
        )
        
        assistant_message = completion.choices[0].message
        logging.info("Summary generated successfully")
        return assistant_message.content
    except Exception as e:
        logging.error(f"Error generating summary: {str(e)}")
        raise

# Add this new function to save summaries
def save_summary(youtube_url, reading_time, summary, video_title):
    history_file = 'summary_history.json'
    timestamp = datetime.now().isoformat()
    summary_data = {
        'youtube_url': youtube_url,
        'video_title': video_title,
        'reading_time': reading_time,
        'summary': summary,
        'timestamp': timestamp
    }
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(summary_data)
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

@app.route('/', methods=['GET', 'POST'])
def index():
    balance_info = check_balance()
    
    if request.method == 'POST':
        youtube_url = request.form['youtube_url']
        reading_time = request.form['reading_time']
        
        logging.info(f"Processing request for URL: {youtube_url}, Reading time: {reading_time}")
        
        try:
            transcript, video_title = download_transcript(youtube_url)
            summary = generate_summary(transcript, reading_time, video_title)
            
            # Save the summary
            save_summary(youtube_url, reading_time, summary, video_title)
            
            # Convert summary to markdown
            summary_html = markdown.markdown(summary)
            
            logging.info("Request processed successfully")
            return render_template('result.html', summary_html=summary_html, transcript=transcript, video_title=video_title)
        except Exception as e:
            logging.error(f"Error processing request: {str(e)}")
            return render_template('index.html', balance_info=balance_info, error=str(e))
    
    return render_template('index.html', balance_info=balance_info)

# Add a new route for the history page
@app.route('/history')
def history():
    history_file = 'summary_history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    return render_template('history.html', history=history)

# Add a new route to delete a summary
@app.route('/delete/<timestamp>')
def delete_summary(timestamp):
    history_file = 'summary_history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        history = [item for item in history if item['timestamp'] != timestamp]
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    return redirect(url_for('history'))

# Modify the download route to handle history downloads
@app.route('/download/<file_type>')
def download(file_type):
    logging.info(f"Downloading {file_type}")
    if file_type == 'summary':
        content = request.args.get('content')
        filename = 'summary.md'
    elif file_type == 'transcript':
        content = request.args.get('content')
        filename = 'transcript.md'
    elif file_type == 'history_summary':
        timestamp = request.args.get('timestamp')
        history_file = 'summary_history.json'
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
            summary_data = next((item for item in history if item['timestamp'] == timestamp), None)
            if summary_data:
                content = summary_data['summary']
                filename = f"summary_{timestamp}.md"
            else:
                return "Summary not found", 404
    else:
        logging.error(f"Invalid file type: {file_type}")
        return "Invalid file type", 400
    
    buffer = io.BytesIO(content.encode('utf-8'))
    buffer.seek(0)
    logging.info(f"File {filename} prepared for download")
    return send_file(buffer, as_attachment=True, download_name=filename, mimetype='text/markdown')

if __name__ == '__main__':
    app.run(debug=True)
