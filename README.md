# **AI Chatbot using LLM & Flask**  

## **Overview**  
This project is a web-based chatbot powered by a **Large Language Model (LLM)**. The backend is built using **Flask**, and the frontend is well-prepared to provide an interactive user experience. The chatbot leverages **Facebook's BlenderBot-400M** model from the **Hugging Face Transformers library** to process user inputs and generate intelligent responses.  

### **Key Features**  
1. Natural Language Processing (NLP) for human-like conversations  
2. Persistent conversation history for better contextual replies  
3. Lightweight backend using Flask  
4. User-friendly web interface  

---

## **1. Introduction**  
Chatbots are widely used in various industries, from customer service to virtual assistants. This project demonstrates how to integrate **LLMs** into a chatbot application using **Flask** as the backend and **Facebook’s BlenderBot** as the AI model. It processes user inputs, maintains conversation history, and generates responses dynamically.


<p align="center">
  <img src="https://github.com/so123-design/AI-Chatbot-using-LLM-Flask/blob/48064b294eb9e393b4fbdc08ae38a83aaf197fe0/LLM%20chatbot%20screenshot%202.PNG" alt="My Image" width="300">
</p>



---

## **2. Technologies Used**  
- **Python** (Backend development)  
- **Flask** (Web framework)  
- **Flask-CORS** (Handling Cross-Origin Resource Sharing)  
- **Hugging Face Transformers** (LLM model processing)  
- **Torch** (Deep learning framework)  
- **HTML, CSS, JavaScript** (Frontend for user interaction)  

---

## **3. Backend Implementation (Flask API)**  
The backend is responsible for handling user inputs, processing responses through the **BlenderBot-400M** model, and maintaining conversation history.  

### **Backend Code**  
```python
from flask import Flask, request, render_template
from flask_cors import CORS
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load pre-trained BlenderBot model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Store conversation history
conversation_history = []

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')  # Serve frontend

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    data = json.loads(request.get_data(as_text=True))
    input_text = data['prompt']
    
    # Format conversation history
    history = "\n".join(conversation_history)

    # Tokenize input text with history for context awareness
    inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")

    # Generate AI response
    outputs = model.generate(**inputs)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Update conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)

    return response

if __name__ == '__main__':
    app.run(debug=True)
```

### **Backend Breakdown**  
- **`Flask` API** handles HTTP requests for the chatbot.  
- **BlenderBot-400M** is used for generating AI-driven conversations.  
- **Conversation history** maintains context across messages.  
- **`POST /chatbot` route** processes user inputs and returns AI responses.  

---
## **4. Frontend Implementation**  
The frontend is built using HTML, CSS, and JavaScript to provide a clean and interactive chat interface. It sends user inputs to the Flask backend via AJAX and displays responses dynamically.
Please check the provided code file about this section for more details

---

## **5. Conclusion**  
This project demonstrates how to build a chatbot using **Flask and a Transformer-based LLM**. It maintains conversation history and generates human-like responses. By combining **Flask for the backend** and **HTML/JavaScript for the frontend**, we achieve a simple yet powerful chatbot system.  
 
---
