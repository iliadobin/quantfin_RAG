"""
Bot configuration and constants.
"""
import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()


class BotConfig:
    """Configuration for Telegram bot."""
    
    # Telegram settings
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    
    # Available pipelines
    PIPELINES: Dict[str, str] = {
        "v1": "RAGv1_Dense",
        "v2": "RAGv2_Hybrid", 
        "v3": "RAGv3_MultiQuery",
        "v4": "RAGv4_ParentChild",
        "v5": "RAGv5_Evidence"
    }
    
    # Available LLM models (DeepSeek)
    LLM_MODELS: Dict[str, str] = {
        "chat": "deepseek-chat",
        "reasoner": "deepseek-reasoner"
    }
    
    # Corpus profiles
    CORPUS_PROFILES: List[str] = ["public"]
    
    # Defaults
    DEFAULT_PIPELINE: str = "v1"
    DEFAULT_MODEL: str = "chat"
    DEFAULT_CORPUS: str = "public"
    DEFAULT_TOP_K: int = 10
    
    # UI settings
    MAX_MESSAGE_LENGTH: int = 4000  # Telegram limit is ~4096
    SHOW_DEBUG_BY_DEFAULT: bool = False
    
    # Data paths
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    INDICES_DIR: str = os.path.join(DATA_DIR, "indices")
    PARSED_DIR: str = os.path.join(DATA_DIR, "parsed")
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration."""
        if not cls.TELEGRAM_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")
        
        if not os.path.exists(cls.DATA_DIR):
            raise ValueError(f"Data directory not found: {cls.DATA_DIR}")
    
    @classmethod
    def get_pipeline_description(cls, pipeline_key: str) -> str:
        """Get human-readable pipeline description."""
        descriptions = {
            "v1": "Dense retrieval (vector search only)",
            "v2": "Hybrid search (BM25 + vectors) with reranking",
            "v3": "Multi-query expansion with fusion",
            "v4": "Parent-child retrieval (context expansion)",
            "v5": "Evidence validation (claim verification)"
        }
        return descriptions.get(pipeline_key, "Unknown pipeline")
    
    @classmethod
    def get_model_description(cls, model_key: str) -> str:
        """Get human-readable model description."""
        descriptions = {
            "chat": "DeepSeek Chat (fast, balanced)",
            "reasoner": "DeepSeek Reasoner (slower, more thorough)"
        }
        return descriptions.get(model_key, "Unknown model")


# Validate on import
if os.getenv("SKIP_CONFIG_VALIDATION") != "1":
    try:
        BotConfig.validate()
    except ValueError as e:
        # Don't fail on import, just warn
        import warnings
        warnings.warn(f"Bot configuration validation failed: {e}")

