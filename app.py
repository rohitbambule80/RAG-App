import streamlit as st
import os
import google.generativeai as genai
from pypdf import PdfReader
import faiss
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Chat with PDF", page_icon="🤖")
st.title("🤖 Chat with your PDF")

# --- Sidebar: API Key & File Upload ---
with st.sidebar:
    st.header("Configuration")
    
    # Option 1: Get key from Streamlit Secrets (Best for deployment)
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        # Option 2: Manual entry (Best for local testing)
        api_key = st.text_input("Enter Gemini API Key", type="password")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# --- Main Logic ---

if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("Please enter your API Key in the sidebar to continue.")
    st.stop()

if uploaded_file is not None:
    # 1. Read PDF
    # We cache this function so it doesn't re-run on every chat message
    @st.cache_resource
    def process_pdf(file):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    # Processing the file
    with st.spinner("Reading PDF..."):
        text = process_pdf(uploaded_file)
        st.success("PDF Loaded!")

    # 2. Smart Chunking (From our previous step)
    @st.cache_resource
    def create_vector_db(text):
        # Chunking
        words = text.split(' ')
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_size = 200 # Adjust as needed

        for word in words:
            if current_length + len(word) + 1 > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Embedding
        embeddings = []
        for chunk in chunks:
            # Simple retry logic for embeddings
            for attempt in range(3):
                try:
                    embed = genai.embed_content(model="models/embedding-001", content=chunk)
                    embeddings.append(embed["embedding"])
                    break
                except:
                    pass # In production, you might want better error handling here
        
        np_embeds = np.array(embeddings).astype("float32")
        dimension = np_embeds.shape[1]
        
        index = faiss.IndexFlatL2(dimension)
        index.add(np_embeds)
        return index, chunks

    with st.spinner("Indexing Document... (This may take a moment)"):
        index, chunks = create_vector_db(text)

    # 3. Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask a question about your PDF..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate Answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # RAG Search
                query_vector = np.array([genai.embed_content(model="models/embedding-001", content=prompt)["embedding"]]).astype("float32")
                D, I = index.search(query_vector, k=3)
                context = "\n".join([chunks[i] for i in I[0]])

                # Gemini Generation
                full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(full_prompt)
                
                st.markdown(response.text)
                
        st.session_state.messages.append({"role": "assistant", "content": response.text})

else:
    st.info("Please upload a PDF to start chatting.")