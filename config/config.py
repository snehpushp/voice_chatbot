from typing import List, Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class TranscriberConfig(BaseSettings):
    model: Literal["distil-whisper-large-v3-en", "whisper-large-v3"]
    language: str = Field(default="en")

    # Speech Recognition Configuration
    energy_threshold: float = Field(default=2000)
    phrase_threshold: float = Field(default=0.1)
    dynamic_energy_threshold: bool = Field(default=True)

    # Audio Settings
    default_timeout: int = Field(default=5)
    audio_sample_rate: int = Field(default=16000)
    audio_channels: int = Field(default=1)
    audio_sample_width: int = Field(default=2)


class RAGConfig(BaseSettings):
    embedding_model: str
    contextual_model: str
    chunk_size: int
    chunk_overlap: int


class ConversationManagerConfig(BaseSettings):
    max_history: int
    model: str


class GroqConfig(BaseSettings):
    rate_limit: int


class DeepgramSpeakerConfig(BaseSettings):
    model: str


class FilesConfig(BaseSettings):
    files: List[str]


class AppConfig(BaseSettings):
    transcriber: TranscriberConfig
    rag: RAGConfig
    conversation_manager: ConversationManagerConfig
    groq: GroqConfig
    deepgram_speaker: DeepgramSpeakerConfig
    files: FilesConfig
    stop_keywords: List[str]


def get_config(config_path: str = "config/config.yaml") -> AppConfig:
    import yaml

    # Load yaml config
    with open(config_path, "r") as file:
        yaml_config = yaml.safe_load(file)

    return AppConfig(**yaml_config)


# Usage
config = get_config()
