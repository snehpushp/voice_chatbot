import asyncio
import os
import re
from collections import deque
from io import BytesIO

import aiohttp
import pyaudio
from loguru import logger
from pydub import AudioSegment


class BackgroundAudioPlayer:
    """
    A class for playing audio files (MP3 or WAV) in the background asynchronously.

    This class uses PyAudio for audio playback and pydub for handling different
    audio formats. It allows for non-blocking audio playback, making it suitable
    for use in applications where other operations need to be performed while
    audio is playing.

    Attributes:
        p (pyaudio.PyAudio): PyAudio instance for audio playback.
        stream (pyaudio.Stream): Audio stream for playback.
        playing (bool): Flag indicating whether audio is currently playing.

    Usage:
        player = BackgroundAudioPlayer()

        # To play an entire audio file:
        await player.play_file('path/to/your/audiofile.mp3')

        # To play a single chunk of audio:
        player.start_stream()
        player.play_in_background(your_audio_chunk)
        # ... do other work ...
        player.stop_stream()
    """

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.playing = False

    def start_stream(self, format=pyaudio.paInt16, channels=2, rate=44100):
        """
        Start the audio stream.

        Args:
            format (int): Audio format (default: pyaudio.paInt16).
            channels (int): Number of audio channels (default: 2).
            rate (int): Sample rate (default: 44100).
        """
        if self.stream is None:
            self.stream = self.p.open(format=format, channels=channels, rate=rate, output=True)

    def stop_stream(self):
        """Stop the audio stream and reset the playing flag."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.playing = False

    async def play_chunk(self, chunk):
        """
        Play a single chunk of audio data.

        Args:
            chunk (bytes): Audio data to play.
        """
        if self.stream and self.stream.is_active():
            self.stream.write(chunk)
        await asyncio.sleep(0)  # Yield control to allow other tasks to run

    def play_in_background(self, chunk):
        """
        Initiate background playback of an audio chunk.

        Args:
            chunk (bytes): Audio data to play.
        """
        if not self.playing:
            self.playing = True
            asyncio.create_task(self._background_player(chunk))

    async def _background_player(self, chunk):
        """
        Internal method to handle background playback of a chunk.

        Args:
            chunk (bytes): Audio data to play.
        """
        await self.play_chunk(chunk)
        self.playing = False


class DeepgramSpeaker:
    """
    An asynchronous class for performing text-to-speech using Deepgram API with low-latency playback.

    This class uses aiohttp for asynchronous API requests and BackgroundAudioPlayer
    for immediate audio playback of received bytes. It optimizes performance by
    fetching and processing the next audio segment while playing the current one.
    """

    def __init__(self, model: str = "aura-helios-en"):
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY environment variable is not set")
        self.url = f"https://api.deepgram.com/v1/speak?model={model}"
        self.headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"}
        self.player = BackgroundAudioPlayer()
        self.player.start_stream()
        self.audio_buffer = deque()
        self.buffer_event = asyncio.Event()
        self.session = None  # Will be initialized in the speak_long_text method
        logger.info("DeepgramSpeaker service started")

    @staticmethod
    def segment_text(text, max_length=2000):
        """
        Segment the input text into chunks, each less than max_length characters,
        first splitting by sentence boundaries and then further splitting long sentences if necessary.

        Args:
            text (str): Input text to be segmented.
            max_length (int): Maximum length of each segment in characters. Default is 2000.

        Returns:
            list: List of segmented chunks, each not exceeding the maximum length.
        """
        # Split the text into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        segments = []
        for sentence in sentences:
            # Skip in case it's empty space
            if not sentence.strip():
                continue

            # If the sentence is already shorter than max_length, add it as is
            if len(sentence) <= max_length:
                segments.append(sentence)
            else:
                # Split long sentences
                while len(sentence) > max_length:
                    split_index = sentence.rfind(" ", 0, max_length)
                    if split_index == -1:  # If no space found, force split at max_length
                        split_index = max_length
                    segments.append(sentence[:split_index].strip())
                    sentence = sentence[split_index:].strip()
                if sentence:  # Add any remaining part of the sentence
                    segments.append(sentence)

        return segments

    async def fetch_audio(self, text):
        """
        Fetch audio data for a given text segment from Deepgram API.

        Args:
            text (str): Text to be converted to speech.

        Returns:
            BytesIO: Buffer containing the audio data, or None if the request failed.
        """
        payload = {"text": text}
        logger.info(f"Sending request to Deepgram API: {text[:50]}...")
        try:
            async with self.session.post(self.url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    return BytesIO(await response.read())
                else:
                    logger.error(f"Error {response.status} from Deepgram API: {await response.text()}")
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"Network error when contacting Deepgram API: {e}")
            return None

    async def process_audio(self, audio_data):
        """
        Process the raw audio data and add it to the audio buffer.

        Args:
            audio_data (BytesIO): Raw audio data from Deepgram API.
        """
        audio_segment = AudioSegment.from_mp3(audio_data).set_frame_rate(44100).set_channels(2).set_sample_width(2)
        raw_data = audio_segment.raw_data
        chunk_size = 1024
        for i in range(0, len(raw_data), chunk_size):
            chunk = raw_data[i : i + chunk_size]
            self.audio_buffer.append(chunk)
        self.buffer_event.set()
        logger.info(f"Processed audio data: {len(raw_data)} bytes")

    async def play_audio(self):
        """
        Continuously play audio chunks from the buffer.
        """
        while True:
            if not self.audio_buffer:
                await self.buffer_event.wait()
                self.buffer_event.clear()

            while self.audio_buffer:
                chunk = self.audio_buffer.popleft()
                self.player.play_in_background(chunk)
                await asyncio.sleep(0.01)  # Adjust this value if needed

            if not self.player.playing:
                await asyncio.sleep(0.1)

    async def speak_long_text(self, text):
        segments = self.segment_text(text)
        logger.info(f"Text segmented into {len(segments)} parts")
        play_task = asyncio.create_task(self.play_audio())

        self.session = aiohttp.ClientSession()
        try:
            next_audio_data = None
            for i, segment in enumerate(segments):
                if next_audio_data:
                    audio_data = next_audio_data
                    next_audio_data = None
                else:
                    audio_data = await self.fetch_audio(segment)

                if audio_data:
                    await self.process_audio(audio_data)
                else:
                    logger.warning(f"Failed to get audio for segment {i+1}")
                    continue

                if i < len(segments) - 1:
                    next_segment = segments[i + 1]
                    next_audio_data_task = asyncio.create_task(self.fetch_audio(next_segment))

                while self.audio_buffer or self.player.playing:
                    await asyncio.sleep(0.1)

                if i < len(segments) - 1:
                    next_audio_data = await next_audio_data_task

            while self.audio_buffer or self.player.playing:
                await asyncio.sleep(0.1)

        finally:
            await self.session.close()
            play_task.cancel()
            try:
                await play_task
            except asyncio.CancelledError:
                pass
            logger.info("Finished processing all segments")

    def close(self):
        self.player.stop_stream()
        logger.info("DeepgramSpeaker service stopped")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    async def main():
        logger.info("Starting the process")
        with open("input_text.txt", "r") as file:
            input_text = file.read()

        speaker = DeepgramSpeaker()
        try:
            await speaker.speak_long_text(input_text)
        finally:
            speaker.close()
        logger.info("Process completed")

    asyncio.run(main())
