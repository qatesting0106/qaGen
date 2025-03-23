import os
import pandas as pd
from typing import List, Optional, Dict
from langchain_community.document_loaders import TextLoader
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_core.prompts import PromptTemplate
import asyncio
import os

# Set HF_TOKEN environment variable for Mistral tokenizer
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")  # Set your Hugging Face token in environment variables

class AsyncMistralAIEmbeddings(MistralAIEmbeddings):
    """Async wrapper for MistralAI embeddings compatible with RAGAS metrics"""
    def __init__(self, mistral_api_key: str):
        super().__init__(
            model="mistral-embed",
            mistral_api_key=mistral_api_key
        )

    def __call__(self, text: str) -> List[float]:
        """Handle direct callable usage for compatibility"""
        return self.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(texts)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.get_event_loop().run_in_executor(
            None, 
            super().embed_documents, 
            texts
        )

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(text)

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.get_event_loop().run_in_executor(
            None,
            super().embed_query,
            text
        )
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
# Import moved to top of file
from langchain_groq import ChatGroq
from giskard.rag import KnowledgeBase
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from langchain.schema import Document

@dataclass
class AgentAnswer:
    message: str
    documents: Optional[List[Document]] = None
    security_assessment: Optional[Dict[str, Any]] = None
from src.core.security_evaluator import SecurityEvaluator

class QAProcessor:
    def __init__(self, mistral_api_key: str, groq_api_key: str):
        self.mistral_api_key = mistral_api_key
        self.groq_api_key = groq_api_key
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.embeddings = None
        self.vector_store = None
        self.llm_model = None
        self.qa_chain = None
        self.security_evaluator = SecurityEvaluator()

    def load_documents(self, file_path) -> List:
        """Load and split documents from the given file path or uploaded file."""
        try:
            # Handle Streamlit UploadedFile
            if hasattr(file_path, 'read'):
                content = file_path.read().decode('utf-8')
                documents = [Document(page_content=content)]
            else:
                # Handle regular file path
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"The file {file_path} does not exist.")
                loader = TextLoader(file_path)
                documents = loader.load()
            
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            raise Exception(f"Error loading document: {str(e)}")

    def initialize_embeddings(self) -> None:
        """Initialize the Mistral AI embeddings model."""
        self.embeddings = AsyncMistralAIEmbeddings(
            mistral_api_key=self.mistral_api_key
        )

    def create_vector_store(self, documents: List) -> None:
        """Create FAISS vector store from documents."""
        if not self.embeddings:
            self.initialize_embeddings()
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def initialize_llm(self) -> None:
        """Initialize the LLM model using GROQ API."""
        self.llm_model = ChatGroq(
            model_name="mistral-saba-24b",
            api_key=self.groq_api_key,
            temperature=0.7
        )

    def create_qa_chain(self) -> None:
        """Create the QA chain with custom prompt template."""
        if not self.llm_model or not self.vector_store:
            raise ValueError("LLM model and vector store must be initialized first.")

        prompt_template = PromptTemplate(
            template="""Answer the following question based only on the provided context:

Context:
{context}

Question:
{question}

Your answer:
""",
            input_variables=["question", "context"]
        )

        self.qa_chain = RetrievalQA.from_llm(
            llm=self.llm_model,
            retriever=self.vector_store.as_retriever(),
            prompt=prompt_template
        )

    def generate_answer(self, question: str, history: Optional[List[Dict]] = None) -> AgentAnswer:
        """Generate an answer using the RAG model."""
        if not self.qa_chain:
            raise ValueError("QA chain is not initialized.")

        messages = history if history else []
        messages.append({"role": "user", "content": question})

        try:
            agent_output = self.qa_chain.invoke({"query": question})["result"]
            answer = str(agent_output)
            documents = getattr(agent_output, 'source_nodes', [])
            
            # Evaluate security risks
            security_assessment = self.security_evaluator.evaluate_security_risks(question, answer)
            
            return AgentAnswer(message=answer, documents=documents, security_assessment=security_assessment)
        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")

    def create_knowledge_base(self, documents: List) -> KnowledgeBase:
        """Create a knowledge base from the provided documents."""
        knowledge_base_df = pd.DataFrame([node.page_content for node in documents], columns=["text"])
        return KnowledgeBase.from_pandas(knowledge_base_df)