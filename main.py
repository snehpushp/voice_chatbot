import asyncio
import uuid

from dotenv import load_dotenv
from loguru import logger

from config.config import config
from src.audio.deepgram_speaker import DeepgramSpeaker
from src.document_processing.extractor import Extractor
from src.session.conversation_manager import ConversationManager
from src.transcription.transcriber import Transcriber

load_dotenv()

# Initialize components using the configuration
conversation_manager = ConversationManager(**config.conversation_manager.dict())
transcriber = Transcriber()
speaker = DeepgramSpeaker(model=config.deepgram_speaker.model)
extractor = Extractor()

# Generate a unique user ID
user_id = str(uuid.uuid4())
logger.info(f"Generated User ID: {user_id}")


async def process_file(file_path: str):
    """Extract content from a file and add it to the RAG system."""
    try:
        content = extractor.extract(file_path)
        if content:
            added_count = await conversation_manager.upload_context([content])
            logger.info(f"Added content from {file_path}. New documents added: {added_count}")
        else:
            logger.error(f"Failed to extract content from {file_path}")
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")


async def process_speech_to_text():
    """Record audio and transcribe it to text."""
    audio_data = await transcriber.record_audio()
    if audio_data:
        transcript, processing_time = await transcriber.transcribe_audio(audio_data)
        if transcript:
            logger.info(f"Transcribed: {transcript}")
            logger.info(f"Transcription time: {processing_time:.2f} seconds")
            return transcript
    return None


async def main():
    conversation_id = user_id  # Use the generated UUID as the conversation ID

    logger.info("Starting voice-based RAG system.")

    # Process files from configuration
    for file_path in config.files.files:
        await process_file(file_path)

    logger.info("File processing complete. You can now start speaking.")
    logger.info(f"The system will automatically exit when you use keywords: {', '.join(config.stop_keywords)}")

    try:
        while True:
            # Process speech
            user_query = await process_speech_to_text()
            if not user_query:
                continue

            # Check for stop keywords
            if any(keyword in user_query.lower() for keyword in config.stop_keywords):
                logger.info("Stop keyword detected. Exiting the conversation...")
                await speaker.speak_long_text("Goodbye! Ending the conversation.")
                break

            # Process query
            result = await conversation_manager.process_query(conversation_id, user_query)
            logger.info(f"AI Response: {result['answer']}")
            logger.info(f"Processing time: {result['processing_time']} seconds")

            # Text to speech for AI response
            await speaker.speak_long_text(result["answer"])

            # Display conversation history (optional)
            history = conversation_manager.get_conversation_history(conversation_id)
            logger.info("Recent Conversation History:")
            for message in history[-5:]:  # Display last 5 messages
                logger.info(f"{message['role'].capitalize()}: {message['content']}")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Exiting...")
    finally:
        # Cleanup
        speaker.close()
        await transcriber.stop_transcribing()


if __name__ == "__main__":
    asyncio.run(main())
