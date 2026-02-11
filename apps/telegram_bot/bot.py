"""
Main Telegram bot application.

Initializes and runs the bot with all handlers.
"""
import logging
import sys
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters
)

from apps.telegram_bot.config import BotConfig
from apps.telegram_bot.state import StateManager
from apps.telegram_bot.handlers import BotHandlers
from apps.telegram_bot.pipeline_factory import PipelineFactory


# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('telegram_bot.log')
    ]
)

logger = logging.getLogger(__name__)


class QAAssistantBot:
    """
    Main bot application.
    
    Sets up handlers and manages bot lifecycle.
    """
    
    def __init__(self):
        """Initialize bot."""
        # Validate configuration
        try:
            BotConfig.validate()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise
        
        # Initialize components
        self.state_manager = StateManager()
        self.pipeline_factory = PipelineFactory(corpus_profile="public")
        
        # Create handlers with factory function
        self.handlers = BotHandlers(
            state_manager=self.state_manager,
            pipeline_factory=self.pipeline_factory.create_pipeline
        )
        
        # Create application
        self.application = (
            Application.builder()
            .token(BotConfig.TELEGRAM_TOKEN)
            .build()
        )
        
        logger.info("QAAssistantBot initialized")
    
    def setup_handlers(self) -> None:
        """Register all handlers."""
        app = self.application
        h = self.handlers
        
        # Command handlers
        app.add_handler(CommandHandler("start", h.start_command))
        app.add_handler(CommandHandler("help", h.help_command))
        app.add_handler(CommandHandler("settings", h.settings_command))
        app.add_handler(CommandHandler("pipeline", h.pipeline_command))
        app.add_handler(CommandHandler("model", h.model_command))
        app.add_handler(CommandHandler("corpus", h.corpus_command))
        app.add_handler(CommandHandler("debug", h.debug_command))
        app.add_handler(CommandHandler("reset", h.reset_command))
        
        # Message handler (for questions)
        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                h.handle_message
            )
        )
        
        # Callback query handler (for buttons)
        app.add_handler(CallbackQueryHandler(h.handle_callback))
        
        # Error handler
        app.add_error_handler(h.error_handler)
        
        logger.info("Handlers registered")
    
    def run(self) -> None:
        """Run the bot."""
        self.setup_handlers()
        
        logger.info("Starting bot polling...")
        
        # Run bot
        self.application.run_polling(
            allowed_updates=['message', 'callback_query']
        )
    
    def stop(self) -> None:
        """Stop the bot gracefully."""
        logger.info("Stopping bot...")
        # Add any cleanup here if needed


def main():
    """Main entry point."""
    try:
        bot = QAAssistantBot()
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

