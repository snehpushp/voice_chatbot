import asyncio
from io import BytesIO
from typing import Optional, Tuple

import speech_recognition as sr
from groq import Groq
from loguru import logger
from pydub import AudioSegment

from config.config import config


class Transcriber:
    def __init__(self):
        self.config = config.transcriber
        self.groq_client = Groq()
        self.is_transcribing = False

        # Setting up SpeechRecognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = self.config.energy_threshold
        self.recognizer.phrase_threshold = self.config.phrase_threshold
        self.recognizer.dynamic_energy_threshold = self.config.dynamic_energy_threshold

    async def process_audio(self, audio_data: sr.AudioData, format: str = "mp3") -> BytesIO:
        wav_data = audio_data.get_wav_data()
        audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
        audio_segment = (
            audio_segment.set_frame_rate(self.config.audio_sample_rate)
            .set_channels(self.config.audio_channels)
            .set_sample_width(self.config.audio_sample_width)
        )

        buffer = BytesIO()
        audio_segment.export(buffer, format=format)
        buffer.seek(0)
        return buffer

    async def record_audio(self, phrase_time_limit: Optional[int] = None) -> Optional[sr.AudioData]:
        with sr.Microphone() as source:
            logger.info("Adjusting for ambient noise...")
            await asyncio.to_thread(self.recognizer.adjust_for_ambient_noise, source, duration=1)
            logger.info("Recording started. Speak now...")
            try:
                audio_data = await asyncio.to_thread(
                    self.recognizer.listen,
                    source,
                    timeout=self.config.default_timeout,
                    phrase_time_limit=phrase_time_limit,
                )
                logger.info("Recording finished.")
                return audio_data
            except sr.WaitTimeoutError:
                logger.warning("No speech detected within the timeout period.")
                return None

    async def transcribe_audio(self, audio_data: sr.AudioData) -> Tuple[Optional[str], float]:
        if not audio_data:
            return None, 0

        buffer = await self.process_audio(audio_data)
        start_time = asyncio.get_event_loop().time()
        try:
            transcription = await asyncio.to_thread(
                self.groq_client.audio.transcriptions.create,
                file=("audio.mp3", buffer),
                model=self.config.model,
                response_format="json",
                language=self.config.language,
                temperature=0.0,
            )
            transcript = transcription.text
        except Exception as e:
            logger.error(f"Error from Groq API: {str(e)}")
            transcript = None
        end_time = asyncio.get_event_loop().time()

        processing_time = end_time - start_time
        return transcript, processing_time

    async def start_transcribing(self):
        self.is_transcribing = True
        while self.is_transcribing:
            audio_data = await self.record_audio()
            if audio_data:
                transcript, processing_time = await self.transcribe_audio(audio_data)
                if transcript:
                    logger.info(f"Transcript: {transcript}")
                    logger.info(f"Processing time: {processing_time:.2f} seconds")
            await asyncio.sleep(0.1)

    async def stop_transcribing(self):
        self.is_transcribing = False


async def main():
    import os

    from dotenv import load_dotenv

    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    transcriber = Transcriber()
    try:
        await transcriber.start_transcribing()
    except KeyboardInterrupt:
        logger.info("Stopping transcription...")
    finally:
        await transcriber.stop_transcribing()


if __name__ == "__main__":
    asyncio.run(main())
