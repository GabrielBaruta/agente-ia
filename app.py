import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai # Biblioteca direta do Google para diagnóstico

# --- CONFIGURAÇÃO DE SEGURANÇA ---
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"]) # Configura diagnóstico
    else:
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # Configura diagnóstico
except Exception as e:
    pass

st.set_page_config(page_title="Lumina | AI Notebook", page_icon="✨", layout="wide")

# Carregar CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except:
    pass

# --- Funções do Backend ---

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(model_name_input):
    prompt_template = """
    Você é o Lumina, um assistente de pesquisa avançado (estilo Notebook LM).
    Responda à pergunta com base no contexto fornecido dos documentos.
    Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos carregados".
    
    Contexto:\n {context}?\n
    Pergunta: \n{question}\n

    Resposta:
    """
    
    # Recupera a chave
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = os.getenv("GOOGLE_API_KEY")

    # Usa o modelo selecionado pelo usuário ou o padrão
    model = ChatGoogleGenerativeAI(model=model_name_input, temperature=0.3, google_api_key=api_key)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, model_choice):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        # Passa a escolha do modelo para a função
        chain = get_conversational_chain(model_choice)
        
        with st.spinner('Lumina está analisando...'):
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            
        return response["output_text"]
    except Exception as e:
        return f"Erro: {str(e)}"

# --- Interface do Usuário (UI) ---

def main():
    with st.sidebar:
        st.markdown("### ⚙️ Configuração")
        
        # --- DIAGNÓSTICO E SELEÇÃO DE MODELO ---
        st.write("Selecione o modelo:")
        
        # Lista padrão de tentativas
        model_options = ["gemini-1.5-flash", "gemini-pro", "gemini-1.0-pro", "gemini-1.5-flash-001"]
        
        # Botão para buscar modelos reais da conta
        if st.button("🛠 Listar Modelos Disponíveis"):
            try:
                available_models = []
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        # Limpa o prefixo 'models/' para ficar mais fácil
                        name = m.name.replace("models/", "")
                        available_models.append(name)
                st.session_state['my_models'] = available_models
                st.success(f"Encontrados: {len(available_models)} modelos")
            except Exception as e:
                st.error(f"Erro ao listar: {e}")

        # Se tivermos modelos encontrados, usamos eles. Se não, usamos a lista padrão.
        if 'my_models' in st.session_state:
            model_options = st.session_state['my_models']

        selected_model = st.selectbox("Modelo Ativo", model_options)
        
        st.markdown("---")
        st.markdown("### 📁 Arquivos")
        pdf_docs = st.file_uploader("Upload PDF", accept_multiple_files=True, type=['pdf'])
        
        if st.button("Processar Documentos"):
            if pdf_docs:
                with st.spinner("Indexando..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Pronto!")
    
    st.markdown("<h1>✨ Lumina AI</h1>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Pergunte..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not os.path.exists("faiss_index"):
            st.error("Carregue os documentos primeiro.")
        else:
            # Passamos o modelo selecionado
            response = user_input(prompt, selected_model)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()
