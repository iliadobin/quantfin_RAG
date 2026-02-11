"""
User state management for the bot.

Tracks per-user settings: pipeline, model, corpus, debug mode.
"""
from typing import Dict, Optional
from dataclasses import dataclass, field
from apps.telegram_bot.config import BotConfig


@dataclass
class UserState:
    """
    State for a single user session.
    
    Tracks user preferences and settings.
    """
    user_id: int
    username: Optional[str] = None
    
    # Settings
    pipeline: str = BotConfig.DEFAULT_PIPELINE
    model: str = BotConfig.DEFAULT_MODEL
    corpus: str = BotConfig.DEFAULT_CORPUS
    show_debug: bool = BotConfig.SHOW_DEBUG_BY_DEFAULT
    top_k: int = BotConfig.DEFAULT_TOP_K
    
    # Stats
    queries_count: int = 0
    last_query: Optional[str] = None
    
    def get_settings_summary(self) -> str:
        """Get formatted settings summary."""
        return (
            f"📋 **Current Settings**\n\n"
            f"Pipeline: `{self.pipeline}` - {BotConfig.get_pipeline_description(self.pipeline)}\n"
            f"Model: `{BotConfig.LLM_MODELS[self.model]}` - {BotConfig.get_model_description(self.model)}\n"
            f"Corpus: `{self.corpus}`\n"
            f"Top-K chunks: `{self.top_k}`\n"
            f"Debug mode: `{'ON' if self.show_debug else 'OFF'}`\n\n"
            f"Queries asked: {self.queries_count}"
        )


class StateManager:
    """
    Manages user states across sessions.
    
    In-memory storage for MVP. Can be extended to persistent storage.
    """
    
    def __init__(self):
        self._states: Dict[int, UserState] = {}
    
    def get_state(self, user_id: int, username: Optional[str] = None) -> UserState:
        """
        Get or create user state.
        
        Args:
            user_id: Telegram user ID
            username: Optional username
            
        Returns:
            User state
        """
        if user_id not in self._states:
            self._states[user_id] = UserState(
                user_id=user_id,
                username=username
            )
        return self._states[user_id]
    
    def update_pipeline(self, user_id: int, pipeline: str) -> None:
        """Update user's pipeline preference."""
        state = self.get_state(user_id)
        state.pipeline = pipeline
    
    def update_model(self, user_id: int, model: str) -> None:
        """Update user's LLM model preference."""
        state = self.get_state(user_id)
        state.model = model
    
    def update_corpus(self, user_id: int, corpus: str) -> None:
        """Update user's corpus preference."""
        state = self.get_state(user_id)
        state.corpus = corpus
    
    def toggle_debug(self, user_id: int) -> bool:
        """Toggle debug mode for user. Returns new state."""
        state = self.get_state(user_id)
        state.show_debug = not state.show_debug
        return state.show_debug
    
    def update_top_k(self, user_id: int, top_k: int) -> None:
        """Update top-k setting."""
        state = self.get_state(user_id)
        state.top_k = max(1, min(50, top_k))  # Clamp to 1-50
    
    def increment_queries(self, user_id: int, query: str) -> None:
        """Increment query counter and update last query."""
        state = self.get_state(user_id)
        state.queries_count += 1
        state.last_query = query
    
    def reset_state(self, user_id: int) -> None:
        """Reset user state to defaults."""
        if user_id in self._states:
            username = self._states[user_id].username
            self._states[user_id] = UserState(
                user_id=user_id,
                username=username
            )
    
    def get_stats(self) -> Dict:
        """Get overall bot statistics."""
        return {
            "total_users": len(self._states),
            "total_queries": sum(s.queries_count for s in self._states.values())
        }

