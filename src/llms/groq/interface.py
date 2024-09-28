import asyncio
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from loguru import logger
from pydantic import Field, PrivateAttr


class GroqLLMManager:
    _instances: Dict[str, "GroqLLMInterface"] = {}

    @classmethod
    def get_instance(cls, model: str, **kwargs) -> "GroqLLMInterface":
        if model not in cls._instances:
            cls._instances[model] = GroqLLMInterface(model=model, **kwargs)
        return cls._instances[model]


class GroqLLMInterface(BaseChatModel):
    """
    A wrapper class for ChatGroq that implements rate limiting.

    This class ensures that no more than 'rate_limit' requests are made within
    any 60-second rolling window. It uses a singleton-like pattern per model.
    """

    model: str
    rate_limit: int = Field(default_factory=lambda: int(os.getenv("GROQ_RATE_LIMIT", "30")))

    _chat_groq: ChatGroq = PrivateAttr()
    _request_timestamps: deque = PrivateAttr()
    _lock: asyncio.Lock = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._chat_groq = ChatGroq(model=self.model)
        self._request_timestamps = deque(maxlen=self.rate_limit)
        self._lock = asyncio.Lock()
        logger.info(f"Initialized GroqLLMInterface for model {self.model}")

    async def _wait_if_needed(self):
        """
        Check if we need to wait before making a new request and wait if necessary.
        """
        async with self._lock:
            current_time = time.time()
            if len(self._request_timestamps) == self.rate_limit:
                elapsed_time = current_time - self._request_timestamps[0]
                if elapsed_time < 60:
                    wait_time = 60 - elapsed_time
                    logger.debug(f"Rate limit reached for model {self.model}. Waiting for {wait_time:.2f} seconds.")
                    await asyncio.sleep(wait_time)

            # Remove timestamps older than 60 seconds
            while self._request_timestamps and current_time - self._request_timestamps[0] > 60:
                self._request_timestamps.popleft()

            self._request_timestamps.append(current_time)
            logger.debug(
                f"Sent request to Groq for model {self.model}. Current queue size: {len(self._request_timestamps)}"
            )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Asynchronously generate a response from the model with rate limiting.

        Args:
            messages (List[BaseMessage]): The input messages for the ChatGroq model.
            stop (Optional[List[str]]): A list of stop sequences for text generation.
            run_manager (Optional[Any]): A run manager for handling the generation process.
            **kwargs: Additional keyword arguments for the generation process.

        Returns:
            Any: The generated response from the ChatGroq model.
        """
        await self._wait_if_needed()
        return await self._chat_groq._agenerate(messages, stop, run_manager, **kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Synchronously generate a response from the model with rate limiting.

        Args:
            messages (List[BaseMessage]): The input messages for the ChatGroq model.
            stop (Optional[List[str]]): A list of stop sequences for text generation.
            run_manager (Optional[Any]): A run manager for handling the generation process.
            **kwargs: Additional keyword arguments for the generation process.

        Returns:
            Any: The generated response from the ChatGroq model.
        """
        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "groq"


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    async def main():
        # These will be the same instance
        groq1 = GroqLLMManager.get_instance("llama3-8b-8192")
        groq2 = GroqLLMManager.get_instance("llama3-8b-8192")
        print(f"groq1 is groq2: {groq1 is groq2}")  # Should print True

        # This will be a different instance
        groq3 = GroqLLMManager.get_instance("llama-3.1-70b-versatile")
        print(f"groq1 is groq3: {groq1 is groq3}")  # Should print False

        # Test rate limiting
        for i in range(35):
            messages = [HumanMessage(content=f"Test query {i}")]
            result = await groq1.agenerate([messages])
            print(f"Query {i} result: {result.generations[0][0].text}")

    asyncio.run(main())
