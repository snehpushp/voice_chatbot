import asyncio
from typing import Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from src.llms.groq.interface import GroqLLMManager


class RAGSystem:
    def __init__(
        self,
        embedding_model: str = "models/text-embedding-004",
        contextual_model: str = "llama3-8b-8192",
        documents: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the RAG system with optional parameters and documents.

        Args:
            embedding_model (str): The model to use for embeddings.
            contextual_model (str): The model to use for contextual chunking.
            documents (List[str], optional): A list of strings, each representing a document.
            chunk_size (int): The size of each chunk for text splitting.
            chunk_overlap (int): The overlap between chunks for text splitting.
        """
        self.vectorstore = Chroma(embedding_function=GoogleGenerativeAIEmbeddings(model=embedding_model))
        self.contextual_llm = GroqLLMManager.get_instance(model=contextual_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        logger.info(
            f"Initialized RAGSystem with embedding model: {embedding_model}, contextual model: {contextual_model}"
        )

        if documents:
            asyncio.run(self.add_documents_async(documents))

    async def contextual_chunking(self, complete_document: str, chunks: List[Document]) -> List[str]:
        prompt = ChatPromptTemplate.from_template(
            """
Whole Document:
<document>
{document}
</document>

Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context in 2-3 lines to situate this chunk within \
the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
            """
        )
        contextual_chain = prompt | self.contextual_llm | StrOutputParser()

        async def process_chunk(chunk):
            try:
                return await contextual_chain.ainvoke(
                    {"document": complete_document, "chunk_content": chunk.page_content}
                )
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                return ""

        contextual_chunks = await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])
        return contextual_chunks

    async def split_documents(self, document_text: str) -> List[Document]:
        chunks = self.text_splitter.create_documents([document_text])
        contextual_split = await self.contextual_chunking(document_text, chunks)
        return [
            Document(page_content=f"{chunk.page_content}\n{contextual_chunk}")
            for chunk, contextual_chunk in zip(chunks, contextual_split)
        ]

    async def _is_unique_document(self, doc: Document) -> bool:
        results = await self.vectorstore.asimilarity_search_with_score(doc.page_content, k=1)
        if not results:
            return True
        return doc.page_content not in results[0][0].page_content

    async def add_documents_async(self, documents: List[str], batch_size: int = 100) -> int:
        total_added = 0
        for document in documents:
            splits = await self.split_documents(document)
            unique_docs = []

            async for doc in tqdm_asyncio(splits, desc="Filtering Documents"):
                if await self._is_unique_document(doc):
                    unique_docs.append(doc)
                    if len(unique_docs) == batch_size:
                        await self.vectorstore.aadd_documents(unique_docs)
                        total_added += len(unique_docs)
                        unique_docs = []

            if unique_docs:
                await self.vectorstore.aadd_documents(unique_docs)
                total_added += len(unique_docs)

        logger.info(f"Added {total_added} new unique documents to the system.")
        return total_added

    def add_documents(self, documents: List[str], batch_size: int = 100) -> int:
        return asyncio.run(self.add_documents_async(documents, batch_size))


if __name__ == "__main__":
    # Example usage:
    from dotenv import load_dotenv

    load_dotenv()

    # Add initial documents
    with open("input_text.txt", "r") as file:
        input_text = file.read()

    # Initialize Rag System
    rag_system = RAGSystem(documents=[input_text])

    async def query_rag(question: str) -> Dict[str, str]:
        try:
            rag_llm = GroqLLMManager.get_instance(model="llama-3.1-70b-versatile")
            prompt = ChatPromptTemplate.from_template(
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                "Use three sentences maximum and keep the answer concise."
                "\nQuestion: {question} \nContext: {context} \nAnswer:"
            )
            rag_chain = (
                {"context": rag_system.vectorstore.as_retriever(), "question": RunnablePassthrough()}
                | prompt
                | rag_llm
                | StrOutputParser()
            )

            start_time = asyncio.get_event_loop().time()
            answer = await rag_chain.ainvoke(question)
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time

            return {"answer": answer, "processing_time": f"{processing_time:.2f}"}
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {"answer": "An error occurred while processing your query.", "processing_time": "N/A"}

    # Test querying
    result = asyncio.run(query_rag("Who is Siddhartha Gautama"))
    print(f"Answer: {result['answer']}")
    print(f"Processing time: {result['processing_time']} seconds")
