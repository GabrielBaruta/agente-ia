import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- CONFIGURAÇÃO DE SEGURANÇA (Importante!) ---
# Isso garante que a chave funcione tanto localmente quanto no Streamlit Cloud
try:
    # Tenta pegar dos Segredos do Streamlit (Nuvem)
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        # Se não achar, tenta pegar do .env (Local)
        from dotenv import load_dotenv
        load_dotenv()
except FileNotFoundError:
    pass # Apenas ignora se não tiver secrets nem .env (vai dar erro depois se não tiver chave)

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

# --- Funções do Backend (Lógica do Notebook LM) ---

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

@st.cache_resource
def get_embeddings_model():
    # Usando embeddings do Google
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_vector_store(text_chunks):
    # Recupera a chave diretamente dos secrets ou ambiente
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = os.getenv("GOOGLE_API_KEY")

    # Passa a chave explicitamente
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
@st.cache_resource
def get_conversational_chain():
    # --- INTEGRAÇÃO DO SEU AGENTE ---
    # Aqui definimos o comportamento do modelo.
    # Você pode substituir este prompt pelo prompt específico do seu agente fornecido.
    
    prompt_template = """
    Você é o Lumina, um assistente de pesquisa avançado (estilo Notebook LM).
    Responda à pergunta com base no contexto fornecido dos documentos.
    Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos carregados",
    mas tente conectar conceitos se possível. Mantenha um tom analítico e prestativo.
    
    Contexto:\n {context}?\n
    Pergunta: \n{question}\n

    Resposta:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    # Recupera a chave novamente
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = os.getenv("GOOGLE_API_KEY")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        
        with st.spinner('Lumina está analisando seus documentos...'):
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            
        return response["output_text"]
    except Exception as e:
        # Imprime o erro no console para ajudar a debugar se falhar de novo
        print(f"Erro detalhado: {e}")
        return "Por favor, faça o upload dos documentos primeiro."

# --- Interface do Usuário (UI) ---

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Base de Conhecimento")
        
        # Simulação de integração com Google Drive (Botão Visual)
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
                with st.spinner("Indexando conteúdo..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Notebook atualizado!")
            else:
                st.warning("Carregue arquivos primeiro.")
        
        st.markdown("---")
        st.markdown("### Especialidade")
        st.markdown("Análise profunda de textos técnicos e criação de sínteses.")

    # Área Principal
    st.markdown("<h1>✨ Lumina <span style='font-size:0.5em; opacity:0.5'>Notebook AI</span></h1>", unsafe_allow_html=True)
    
    # Exibir status se houver docs processados
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
        # Adiciona pergunta ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Processa resposta
        response = user_input(prompt)
        
        # Adiciona resposta ao histórico
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
