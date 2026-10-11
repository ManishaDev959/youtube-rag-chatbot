
import streamlit as st


st.title("YouTube Chatbot")
st.write("Loading libraries...")


from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# st.title("YouTube Chatbot")

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
    return embeddings, llm

with st.spinner("Loading models (first time takes a minute)..."):
    embeddings, llm = load_models()

url = st.text_input("YouTube URL")

if st.button("Load video"):
    try:
        video_id = url.split("v=")[1].split("&")[0]
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript = " ".join(s.text for s in fetched)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(transcript)

        st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
        st.success(f"Video loaded ({len(chunks)} chunks)")
    except Exception as e:
        st.error(str(e)[:300])

if "vector_store" in st.session_state:
    question = st.text_input("Ask a question about the video")

    if question:
        try:
            docs = st.session_state.vector_store.similarity_search(question, k=4)
            context = "\n\n".join(d.page_content for d in docs)

            prompt = f"""You are answering questions about a YouTube video using its transcript.
Use the transcript below to answer. If it has nothing relevant, say you don't know.

Transcript:
{context}

Question: {question}"""

            st.write(llm.invoke(prompt).text)
        except Exception as e:
            st.error(str(e)[:300])