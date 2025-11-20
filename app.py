# app.py
# -*- coding: utf-8 -*-
"""RAG Web App - improved error handling for embeddings / faiss"""
import streamlit as st
import os
import google.generativeai as genai
from pypdf import PdfReader
import faiss
import numpy as np
import time
import traceback

# --- Page Configuration ---
st.set_page_config(page_title="Chat with PDF", page_icon="🤖")
st.title("🤖 Chat with your PDF")

# --- Sidebar: API Key & File Upload ---
with st.sidebar:
    st.header("Configuration")

    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        api_key = st.text_input("Enter Gemini API Key", type="password")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("Please enter your API Key in the sidebar to continue.")
    st.stop()

if uploaded_file is not None:
    # 1. Read PDF
    @st.cache_resource
    def process_pdf(file):
        reader = PdfReader(file)
        text_parts = []
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            text_parts.append(page_text)
        return "\n".join(text_parts)

    with st.spinner("Reading PDF..."):
        text = process_pdf(uploaded_file)
        st.success("PDF Loaded!")

    # 2. Smart Chunking + Embedding + FAISS Index
    @st.cache_resource
    def create_vector_db(text, chunk_size=200, max_retries=3, retry_delay=1.0):
        # Chunking by words to approx chunk_size characters (simple)
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

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

        if len(chunks) == 0:
            raise ValueError("No text chunks were created from the PDF. The PDF may be empty or unreadable.")

        # Embedding
        embeddings = []
        failed_chunks = []
        for idx, chunk in enumerate(chunks):
            success = False
            last_exception = None
            for attempt in range(max_retries):
                try:
                    # call embed_content
                    resp = genai.embed_content(model="models/embedding-001", content=chunk)
                    # resp could be a dict like {"embedding": [...]} or more complex.
                    # Try to find the embedding in common locations.
                    if isinstance(resp, dict):
                        embed_vec = resp.get("embedding") or resp.get("embeddings") or resp.get("result")
                        # If it's nested, try common patterns:
                        if embed_vec is None and "data" in resp:
                            # some APIs return {"data": [{"embedding": [...]}]}
                            d = resp.get("data")
                            if isinstance(d, (list, tuple)) and len(d) > 0 and isinstance(d[0], dict):
                                embed_vec = d[0].get("embedding") or d[0].get("vector")
                    else:
                        embed_vec = None

                    # If still not found, try attribute access (defensive)
                    if embed_vec is None and hasattr(resp, "embedding"):
                        embed_vec = getattr(resp, "embedding")

                    if embed_vec is None:
                        # try if resp directly is a list/ndarray
                        if isinstance(resp, (list, tuple, np.ndarray)):
                            embed_vec = resp

                    if embed_vec is None:
                        raise RuntimeError(f"Couldn't extract embedding from response for chunk {idx}. Response: {resp}")

                    # Ensure it's a list/1D numeric array
                    arr = np.array(embed_vec, dtype="float32")
                    if arr.ndim != 1:
                        # flatten if nested
                        arr = arr.reshape(-1)

                    embeddings.append(arr)
                    success = True
                    break
                except Exception as e:
                    last_exception = e
                    # small backoff
                    time.sleep(retry_delay * (attempt + 1))
            if not success:
                failed_chunks.append((idx, str(last_exception)))
                # continue without this chunk (or you could choose to stop)
                st.warning(f"Embedding failed for chunk {idx}; it will be skipped. Error: {last_exception}")

        if len(embeddings) == 0:
            # Nothing to index; raise a clear error for the user
            raise RuntimeError(
                "Failed to generate any embeddings. Possible causes: invalid API key, rate limits, model name mismatch, or network issues."
                " Check your API key and model name, and inspect the logs above for more details."
            )

        # Align embeddings into a 2D numpy array: (n_vectors, dim)
        # ensure all embeddings share same dimension
        dims = [e.shape[0] for e in embeddings]
        if len(set(dims)) != 1:
            raise RuntimeError(f"Embeddings have inconsistent dimensions: {set(dims)}. First few dims: {dims[:10]}")

        np_embeds = np.stack(embeddings).astype("float32")  # shape (n, dim)
        n_vectors, dimension = np_embeds.shape

        if dimension <= 0:
            raise RuntimeError(f"Invalid embedding dimension: {dimension}")

        # Build FAISS index
        index = faiss.IndexFlatL2(dimension)
        index.add(np_embeds)

        # Map: because we may have skipped failed chunks, build a filtered chunk list
        valid_chunks = [chunks[i] for i, _ in enumerate(chunks) if not any(i == f_idx for f_idx, _ in failed_chunks)]
        # Note: above enumeration assumes failed_chunks refer to original chunk indices;
        # if large numbers of failures occur, you might want to track indexes when appending embeddings.

        return index, np_embeds, valid_chunks, failed_chunks

    with st.spinner("Indexing Document... (This may take a moment)"):
        try:
            index, np_embeds, chunks, failed_chunks = create_vector_db(text)
            st.success(f"Indexed {np_embeds.shape[0]} vectors (dim={np_embeds.shape[1]}).")
            if failed_chunks:
                st.info(f"Note: embeddings failed for {len(failed_chunks)} chunk(s). They were skipped.")
        except Exception as e:
            st.error("Failed to create vector DB: " + str(e))
            st.exception(traceback.format_exc())
            st.stop()

    # 3. Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask a question about your PDF..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Create query vector - same extraction logic as above
                    resp = genai.embed_content(model="models/embedding-001", content=prompt)
                    if isinstance(resp, dict):
                        q_embed = resp.get("embedding") or resp.get("embeddings")
                        if q_embed is None and "data" in resp:
                            d = resp.get("data")
                            if isinstance(d, (list, tuple)) and len(d) > 0 and isinstance(d[0], dict):
                                q_embed = d[0].get("embedding")
                    else:
                        q_embed = None
                    if q_embed is None and hasattr(resp, "embedding"):
                        q_embed = getattr(resp, "embedding")
                    if q_embed is None and isinstance(resp, (list, tuple, np.ndarray)):
                        q_embed = resp

                    if q_embed is None:
                        raise RuntimeError(f"Couldn't extract embedding for the query. Response: {resp}")

                    query_vector = np.array(q_embed, dtype="float32").reshape(1, -1)

                    # Search
                    k = min(3, index.ntotal) if hasattr(index, "ntotal") else 3
                    D, I = index.search(query_vector, k=k)
                    # I is indices into the vectors we added; find corresponding chunks
                    # We used a filtered chunks list when building the index.
                    retrieved_chunks = []
                    for idx in I[0]:
                        if idx < len(chunks):
                            retrieved_chunks.append(chunks[idx])
                    context = "\n".join(retrieved_chunks)

                    # Gemini Generation
                    full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    response = model.generate_content(full_prompt)

                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error("Error while answering: " + str(e))
                    st.exception(traceback.format_exc())

else:
    st.info("Please upload a PDF to start chatting.")
