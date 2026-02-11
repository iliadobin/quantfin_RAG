"""
Inline keyboards for bot UI.

Provides interactive buttons for selecting pipeline, model, corpus, etc.
"""
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from apps.telegram_bot.config import BotConfig


def get_main_menu_keyboard() -> InlineKeyboardMarkup:
    """Get main menu keyboard."""
    keyboard = [
        [
            InlineKeyboardButton("🔧 Pipeline", callback_data="menu_pipeline"),
            InlineKeyboardButton("🤖 Model", callback_data="menu_model")
        ],
        [
            InlineKeyboardButton("📚 Corpus", callback_data="menu_corpus"),
            InlineKeyboardButton("🐛 Debug", callback_data="toggle_debug")
        ],
        [
            InlineKeyboardButton("📊 Settings", callback_data="show_settings"),
            InlineKeyboardButton("❓ Help", callback_data="show_help")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_pipeline_keyboard(current: str) -> InlineKeyboardMarkup:
    """
    Get pipeline selection keyboard.
    
    Args:
        current: Currently selected pipeline key
        
    Returns:
        Inline keyboard markup
    """
    keyboard = []
    
    # Group pipelines in rows of 2
    pipeline_items = list(BotConfig.PIPELINES.items())
    for i in range(0, len(pipeline_items), 2):
        row = []
        for key, name in pipeline_items[i:i+2]:
            # Add checkmark to current selection
            label = f"✓ {key.upper()}" if key == current else key.upper()
            row.append(
                InlineKeyboardButton(
                    label,
                    callback_data=f"pipeline_{key}"
                )
            )
        keyboard.append(row)
    
    # Add back button
    keyboard.append([InlineKeyboardButton("« Back", callback_data="menu_main")])
    
    return InlineKeyboardMarkup(keyboard)


def get_model_keyboard(current: str) -> InlineKeyboardMarkup:
    """
    Get LLM model selection keyboard.
    
    Args:
        current: Currently selected model key
        
    Returns:
        Inline keyboard markup
    """
    keyboard = []
    
    for key, name in BotConfig.LLM_MODELS.items():
        label = f"✓ {name}" if key == current else name
        keyboard.append([
            InlineKeyboardButton(
                label,
                callback_data=f"model_{key}"
            )
        ])
    
    keyboard.append([InlineKeyboardButton("« Back", callback_data="menu_main")])
    
    return InlineKeyboardMarkup(keyboard)


def get_corpus_keyboard(current: str) -> InlineKeyboardMarkup:
    """
    Get corpus selection keyboard.
    
    Args:
        current: Currently selected corpus
        
    Returns:
        Inline keyboard markup
    """
    keyboard = []
    
    for corpus in BotConfig.CORPUS_PROFILES:
        label = f"✓ {corpus}" if corpus == current else corpus
        keyboard.append([
            InlineKeyboardButton(
                label,
                callback_data=f"corpus_{corpus}"
            )
        ])
    
    keyboard.append([InlineKeyboardButton("« Back", callback_data="menu_main")])
    
    return InlineKeyboardMarkup(keyboard)


def get_citation_expand_keyboard(citation_index: int) -> InlineKeyboardMarkup:
    """
    Get keyboard for expanding/viewing citation details.
    
    Args:
        citation_index: Index of citation (0-based)
        
    Returns:
        Inline keyboard markup
    """
    keyboard = [
        [
            InlineKeyboardButton(
                "📄 View Full Quote",
                callback_data=f"cite_view_{citation_index}"
            )
        ],
        [
            InlineKeyboardButton(
                "📍 View Context",
                callback_data=f"cite_context_{citation_index}"
            )
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_debug_keyboard(answer_id: str) -> InlineKeyboardMarkup:
    """
    Get keyboard for retrieval debug information.
    
    Args:
        answer_id: Unique identifier for the answer
        
    Returns:
        Inline keyboard markup
    """
    keyboard = [
        [
            InlineKeyboardButton(
                "🔍 Show Retrieved Chunks",
                callback_data=f"debug_chunks_{answer_id}"
            )
        ],
        [
            InlineKeyboardButton(
                "⏱️ Show Timing",
                callback_data=f"debug_timing_{answer_id}"
            )
        ],
        [
            InlineKeyboardButton(
                "📋 Show Full Trace",
                callback_data=f"debug_trace_{answer_id}"
            )
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_close_keyboard() -> InlineKeyboardMarkup:
    """Get simple close button."""
    keyboard = [[InlineKeyboardButton("✖️ Close", callback_data="close")]]
    return InlineKeyboardMarkup(keyboard)

