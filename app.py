import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os


# Sidebar contents
with st.sidebar:
    st.title("Policy-QA bot App")

    # Accept OPENAI_API_KEY from the sidebar
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Set OPENAI_API_KEY if provided
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

#load_dotenv()

def main():
    st.header("Policy QA Bot")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Chunk text by page instead of chunk size
        chunks =  [text.strip() for text in text.split("\n\n")]  # Split by page breaks

        #st.write(chunks[0:1])

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user questions/query
        query = st.text_area("Ask questions about your PDF file:", height=100)
        # Checkbox to show source documents
        show_source = st.checkbox("Show source documents?", value=False)

        if query:
            # Perform similarity search
            docs = VectorStore.similarity_search(query=query, k=5)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)

            # Use beta_columns to create two columns
            if show_source:
                col1, col2 = st.columns(2)

            # Display answer in the first column
                with col1:
                    st.subheader("Answer:")
                    st.write(response)

            # Display source documents in the second column if checkbox is selected
                with col2:
                    st.subheader("Source Documents:")
                    st.text_area("Document", docs, height=325)
            else:
                st.subheader("Answer:")
                st.write(response)
                

if __name__ == "__main__":
    main()