---
title: "LangChain RAG Chat"
emoji: "💬"
colorFrom: "blue"
colorTo: "green"
sdk: "gradio"
sdk_version: "3.0"
python_version: "3.10"
app_file: "run_fast_api.py"
fullWidth: true
header: "default"
short_description: "AI-powered chatbot for cocktail recommendations."
models: 
  - "openai/gpt-3.5-turbo"
datasets: 
  - "cocktail_data/cocktails.csv"
tags:
  - "RAG"
  - "LangChain"
  - "Chatbot"
pinned: true
---


# Cocktail RAG Chat

**Cocktail RAG Chat** is an intelligent chatbot that recommends cocktail recipes based on the ingredients entered by the user. The project is built using **Retrieval-Augmented Generation (RAG)**, which effectively combines a **local LLM model** with a **vector database** for searching and recommendations.

### Fully working deployed model: [RAG Cocktail Chat](https://huggingface.co/spaces/stkrk/langchain_rag_chat)

---

## ⚙️ Key Technologies Used:

- **FastAPI**: for creating the web server and handling user queries.
- **LlamaIndex** (formerly GPT Index): for building a **vector index** based on cocktail data (loaded from CSV files).
- **TinyLlama (gguf)**: a local **language model** for generating responses to queries.
- **LangChain**: for building and managing the pipeline with the LLM.
- **Docker**: for containerizing the project and enabling easy deployment.
- **Python (3.10)**: primary programming language.
- **Pandas**: for processing and manipulating the cocktail data (CSV).

---

## 🎬 How the Project Works:

### 1. **Loading Data & Indexing**:
The project uses a **cocktails.csv** file, which contains cocktail recipes with the following attributes:
- Cocktail name
- Ingredients
- Instructions
- Category, glass type, alcohol content

Data is loaded using **Pandas** and passed to **LlamaIndex** to build a **vector index**. This allows efficient querying of relevant recipes for the user.
LlamaIndex (formerly known as GPT Index) is used for creating and querying the vector index of cocktail data. This library converts text data (e.g., cocktail recipes) into vector representations, allowing fast and efficient similarity-based searches across the database.

We are using a custom embedding model, based on **intfloat/e5-small-v2** from HuggingFace. 
This embedder transforms the text into vector representations, which are then used for querying 
within the LlamaIndex vector store. The custom embedder is fine-tuned for fast and accurate 
text processing tailored to the needs of the project, optimizing the performance in a local 
environment without relying on external APIs.

### 2. **Interactive Chat**:
Users can input queries such as:
- *"Do you know cocktails with gin and lime?"*
- *"Can you suggest a cocktail with my favorite ingredients?"*

The query is passed to the **RAGEngine**, which performs:
- Vector search using **LlamaIndex**.
- Response generation using **TinyLlama** (GPT-style model).

**TinyLlama** generates answers based on the retrieved context. If the context is empty or irrelevant, the system returns "Sorry, I don't know" for accuracy.

### 3. **User Memory**:
The **UserMemory** class is responsible for storing and managing the user's preferences, 
such as their favorite ingredients. It allows the system to provide personalized cocktail 
recommendations based on the user's input over time.
How it works:

- **Adding Ingredients:** When the user mentions ingredients they like, the system stores them in memory. 
For example, if the user says **"I like mint and lime"**, the system will store those ingredients and use them in future queries.
- **Retrieving Favorites:** The system can use the stored ingredients to generate recommendations based on the user's preferences. If the user asks for a cocktail with their favorite ingredients, the system will filter the available recipes and suggest the most relevant ones.
- **Clearing Memory:** Users can clear their preferences at any time, resetting the stored data. This allows the system to adapt to changing tastes or preferences.

This mechanism is integrated into the system and allows for a more personalized and user-centric experience.

### 4. **Frontend**:
The interface is built with **HTML + CSS**. For each user query, the chat history is displayed with previous queries and answers. The **dark mode** is included for comfortable use at night.

---

## 📝 Key Decisions Made:

1. **Using Retrieval-Augmented Generation (RAG)**:
   Using **LlamaIndex** for context retrieval greatly improves **response accuracy** compared to purely generative models, where there could be more "hallucinations."

2. **Local LLM**:
   Instead of using external APIs, we utilize **TinyLlama** with **gguf** formats, allowing for offline operation and full control over the model.

3. **Temperature (temperature=0.3)**:
   A low temperature ensures **stable, accurate responses**, avoiding excessive creativity or random generation.

4. **Docker Integration**:
   For easy deployment and scalability, the project is containerized with **Docker**, allowing for seamless execution in any containerized environment.

5. **Vectorization and Search**:
   By using **LlamaIndex**, we can efficiently search for relevant cocktails based on vector representations of the text, improving search results compared to traditional text-based searching.

---

## 📊 Results:

- **Search Performance**: The model retrieves up to **5 relevant cocktails** per query.
- **Response Accuracy**: It finds appropriate recipes from the context without generating new cocktails.
- **Response Speed**: Due to containerization and parameter optimization, responses are generated in **1–3 seconds** (depending on query complexity).
- **Scalability**: The project is ready for **scaling** — adding new cocktails, switching LLM models, integrating new tools.

---
### Structure of project
```
chat_nlp_rag/ 
│
├── data/ 
│ └── cocktails.csv                                     # Cocktail dataset for training 
│
├── models/                                             # Model files (not included in repo)
│ └── tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf 
│
├── rag_pipeline/                                       # Core logic of the RAG pipeline 
│ ├── init.py 
│ ├── custom_embedder.py                                # Custom embedding logic 
│ ├── data_loader.py                                    # Data loading logic 
│ ├── llm_interface.py                                  # Interface with Llama model 
│ ├── rag_engine.py                                     # RAG engine logic 
│ ├── user_memory.py                                    # Memory management for users 
│ └── vector_store.py                                   # Vector store logic for fast retrieval 
│
├── static/                                             # Static files (e.g., favicon) 
│ └── favicon.png 
│
├── templates/                                          # HTML templates for rendering 
│ └── chat.html 
│
├── storage/                                            # Data storage for vector store 
│ └── vector_store_files.json 
│
├── venv/                                               # Virtual environment (not included in repo)
├── .gitignore                                          # Files to ignore in Git 
├── Dockerfile                                          # Docker configuration for deployment 
├── requirements.txt                                    # Python dependencies 
├── run_fast_api.py                                     # FastAPI server entry point 
├── run_local.py                                        # Local development script 
└── space.yaml                                          # Hugging Face Space configuration
```
---

## 🛠️ How to Run:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cocktail-chat.git

2. Create a virtual environment:
    ```bash
   python -m venv venv

3. Install dependencies:
    ```bash
    pip install -r requirements.txt

4. Download the model and save it to the folder "models":
   [Download TinyLlama_Q5](https://huggingface.co/stkrk/tinyllama-gguf/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf?download=true)
<br>
<br>
5. Run the server:
    ```bash
    uvicorn run_fast_api:app --reload

6. Open in your browser:
   
    http://127.0.0.1:8000

### Running the Pipeline Locally (Without FastAPI):

- If you'd prefer to run locally without the FastAPI server, you can use the run_local.py script. 
This will run the RAG pipeline and allow you to interact with the system through the command line.
Make sure all dependencies are installed (as above).

    ```bash
    python run_local.py
---

### Future Plans:

- Expand the cocktail database (adding new recipes).
- Improve query accuracy using more complex models (e.g., Mistral 7B).
- Integrate with voice assistants for a more interactive experience.