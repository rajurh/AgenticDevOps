import os
import requests
import streamlit as st

API_BASE = os.getenv("RAG_BASE", "http://127.0.0.1:8001")

st.set_page_config(page_title="DevOps Knowledge Assistant", page_icon="🚀", layout="wide")
st.title("🚀 DevOps Knowledge Assistant")
st.markdown("*Ask questions about deployment, CI/CD, security, and Azure operations*")

with st.sidebar:
    st.header("⚙️ Settings")
    api_base = st.text_input("Backend API URL", value=API_BASE)
    if api_base != API_BASE:
        API_BASE = api_base
    
    st.markdown("---")
    st.subheader("📝 Sample Questions")
    st.markdown("""
    - What is our release process?
    - How do we handle incidents?
    - What are Azure deployment best practices?
    - What security checks are required?
    - Describe our CI/CD pipeline
    """)

query = st.text_area("💬 Ask your question:", placeholder="e.g., What is our release process?", height=100)

if st.button("🔍 Ask", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching knowledge base..."):
            try:
                resp = requests.post(f"{API_BASE}/api/query", json={"query": query}, timeout=120)
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to contact backend: {e}")
                data = None
            except ValueError:
                st.error("Backend returned an invalid JSON response.")
                data = None

        if data:
            st.success("✅ Answer generated")
            st.markdown("### 💡 Answer")
            st.markdown(data.get("answer", "No answer provided"))
            
            st.markdown("---")
            st.markdown("### 📚 Source Documents")
            retrieved = data.get("retrieved", [])
            if retrieved:
                for i, d in enumerate(retrieved):
                    with st.expander(f"📄 Source {i+1}: {d.get('metadata', {}).get('title', d.get('id'))} (Relevance: {d.get('score', 0):.2%})"):
                        st.caption(f"**Source:** {d.get('metadata', {}).get('source', 'Unknown')}")
                        st.write(d.get("text", ""))
            else:
                st.info("No relevant documents found.")

st.sidebar.markdown("---")
st.sidebar.caption("💡 Tip: Make sure your FastAPI backend is running on the configured URL.")

