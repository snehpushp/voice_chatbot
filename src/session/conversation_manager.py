import asyncio
from typing import Dict, List, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
from pydantic import BaseModel, Field

from src.llms.groq.interface import GroqLLMManager
from src.rag.rag_system import RAGSystem


class Message(BaseModel):
    role: Literal["user", "ai"]
    content: str


class Conversation(BaseModel):
    messages: List[Message] = Field(default_factory=list)


class ConversationManager:
    def __init__(self, model: str = "llama-3.1-70b-versatile", max_history: int = 10):
        """
        Initialize the ConversationManager.

        Args:
            model (str): The model to use for generating responses.
            max_history (int): Maximum number of previous messages to include in the context.
        """
        self.rag_system = RAGSystem()
        self.llm = GroqLLMManager.get_instance(model=model)
        self.max_history = max_history
        self.conversations: Dict[str, Conversation] = {}
        logger.info(f"Initialized ConversationManager with model: {model}, max_history: {max_history}")

    def _get_or_create_conversation(self, conversation_id: str) -> Conversation:
        """Get an existing conversation or create a new one if it doesn't exist."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = Conversation()
        return self.conversations[conversation_id]

    def add_message(self, conversation_id: str, role: Literal["system", "user", "ai"], content: str):
        """Add a message to the conversation history."""
        conversation = self._get_or_create_conversation(conversation_id)
        conversation.messages.append(Message(role=role, content=content))

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get the conversation history for the given conversation ID."""
        conversation = self._get_or_create_conversation(conversation_id)
        return [{"role": msg.role, "content": msg.content} for msg in conversation.messages[-self.max_history :]]

    async def upload_context(self, documents: List[str]) -> int:
        """
        Upload documents to the RAG system for context retrieval.

        Args:
            documents (List[str]): A list of document contents to be added to the RAG system.

        Returns:
            int: The number of new unique documents added to the system.
        """
        try:
            added_count = await self.rag_system.add_documents_async(documents)
            logger.info(f"Successfully added {added_count} new unique documents to the RAG system.")
            return added_count
        except Exception as e:
            logger.error(f"Error uploading documents to RAG system: {str(e)}")
            return 0

    async def process_query(self, conversation_id: str, query: str) -> Dict[str, str]:
        """
        Process a query within the context of a conversation.

        Args:
            conversation_id (str): The ID of the conversation.
            query (str): The user's query.

        Returns:
            Dict[str, str]: A dictionary containing the answer and processing time.
        """
        try:
            # Add the user's query to the conversation history
            self.add_message(conversation_id, "user", query)

            # Retrieve conversation history
            history = self.get_conversation_history(conversation_id)

            # Prepare the prompt template
            chat_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=(
                            "You are an AI assistant engaging in a conversation. "
                            "Use the following pieces of context to answer the user's question. "
                            "If you don't know the answer, just say that you don't know. "
                            "Keep the answer concise and relevant to the conversation."
                        )
                    ),
                    *[
                        AIMessage(content=msg["content"])
                        if msg["role"] == "ai"
                        else HumanMessage(content=msg["content"])
                        for msg in history[:-1]
                    ],  # Exclude the latest query
                ]
            )
            context_prompt = {
                "context": self.rag_system.vectorstore.as_retriever(),
                "query": RunnablePassthrough(),
            } | PromptTemplate.from_template("""Context: {context}\n\nQuestion: {query}""")

            chat_prompt.extend(context_prompt.invoke(query).to_messages())

            # Set up the RAG chain
            rag_chain = chat_prompt | self.llm | StrOutputParser()

            # Process the query
            start_time = asyncio.get_event_loop().time()
            answer = await rag_chain.ainvoke({"query": query})
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time

            # Add the AI's response to the conversation history
            self.add_message(conversation_id, "ai", answer)

            return {"answer": answer, "processing_time": f"{processing_time:.2f}"}
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {"answer": "An error occurred while processing your query.", "processing_time": "N/A"}

    def clear_conversation(self, conversation_id: str):
        """Clear the conversation history for a specific conversation ID."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id].messages.clear()
            logger.info(f"Cleared conversation history for conversation ID: {conversation_id}")

    def delete_conversation(self, conversation_id: str):
        """Delete a conversation and its associated data."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Deleted conversation with ID: {conversation_id}")


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    async def main():
        # Initialize ConversationManager
        conversation_manager = ConversationManager()

        # Upload some context
        documents = [
            "Paris is the capital and most populous city of France. It is located on the Seine River.",
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. It is named after engineer Gustave Eiffel.",
            "The Louvre is the world's largest art museum and a historic monument in Paris, France.",
        ]
        added_count = await conversation_manager.upload_context(documents)
        print(f"Added {added_count} documents to the RAG system.")

        conversation_id = "user123"  # This could be a unique identifier for each user or conversation

        # Process a series of queries
        queries = [
            "What is the capital of France?",
            "Tell me more about its history.",
            "What's a famous landmark there?",
            "Are there any museum",
        ]

        for query in queries:
            result = await conversation_manager.process_query(conversation_id, query)
            print(f"User: {query}")
            print(f"AI: {result['answer']}")
            print(f"Processing time: {result['processing_time']} seconds\n")

        # Get conversation history
        history = conversation_manager.get_conversation_history(conversation_id)
        print("Conversation History:")
        for message in history:
            print(f"{message['role'].capitalize()}: {message['content']}")

    asyncio.run(main())
