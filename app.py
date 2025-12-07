import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Trocamos o embedding do Google pelo HuggingFace (Local e Gratuito)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# --- CONFIGURAÇÃO DE SEGURANÇA ---
# Tenta pegar a chave dos Segredos (Nuvem) ou do .env (Local)
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        load_dotenv()
except FileNotFoundError:
    pass

# --- Configuração da Página ---
st.set_page_config(
    page_title="Lumina | AI Notebook",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carregar CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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
    # MUDANÇA CRUCIAL: Usando modelo local (all-MiniLM-L6-v2)
    # Isso evita o erro de API Key na hora de criar o banco de dados
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Você é o Lumina, um assistente de pesquisa avançado (estilo Notebook LM).
    Responda à pergunta com base no contexto fornecido dos documentos.
    Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos carregados",
    mas tente conectar conceitos se possível. Mantenha um tom analítico e prestativo.
    
    Contexto:\n {context}?\n
    Pergunta: \n{question}\n

    Resposta:
    """
    
    # Recupera a chave de segurança novamente
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = os.getenv("GOOGLE_API_KEY")

    # ATUALIZAÇÃO: Usando 'gemini-1.5-flash' que é mais moderno e estável
    # Passamos a api_key explicitamente para evitar erros de autenticação
    # model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    # Mude de "gemini-1.5-flash" para "gemini-1.5-flash-latest"
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    # Precisamos usar o MESMO modelo de embeddings para ler o banco
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        # allow_dangerous_deserialization é necessário para versões novas do FAISS
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        
        with st.spinner('Lumina está analisando seus documentos...'):
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            
        return response["output_text"]
    except Exception as e:
        return f"Erro ao processar. Detalhes: {str(e)}"

# --- Interface do Usuário (UI) ---

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Base de Conhecimento")
        
        # Simulação de integração com Google Drive
        if st.button("🔄 Conectar Google Drive"):
            st.info("Integração com Drive pronta para configuração de OAuth2.")
        
        st.markdown("---")
        
        pdf_docs = st.file_uploader(
            "Carregue seus PDFs locais", 
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("Processar Documentos"):
            if pdf_docs:
                with st.spinner("Indexando conteúdo com IA local..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Notebook atualizado e pronto!")
            else:
                st.warning("Carregue arquivos primeiro.")
        
        st.markdown("---")
        st.markdown("### Especialidade")
        st.markdown("Análise profunda de textos técnicos.")

    # Área Principal
    st.markdown("<h1>✨ Lumina <span style='font-size:0.5em; opacity:0.5'>Notebook AI</span></h1>", unsafe_allow_html=True)
    
    # Exibir status
    if os.path.exists("faiss_index"):
        st.markdown("""
        <div class="doc-card">
            <span>📚</span>
            <span>Base de conhecimento ativa e indexada. Pergunte sobre seus documentos.</span>
        </div>
        """, unsafe_allow_html=True)

    # Histórico de Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do Usuário
    if prompt := st.chat_input("Faça uma pergunta sobre seus documentos..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Se não houver índice ainda, avisa o usuário
        if not os.path.exists("faiss_index"):
            st.error("Por favor, carregue e processe os documentos primeiro na barra lateral.")
        else:
            response = user_input(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()
