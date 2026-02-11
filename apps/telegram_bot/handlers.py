"""
Handlers for Telegram bot commands and messages.

Processes user interactions: commands, messages, callbacks.
"""
import logging
from typing import Dict, Any
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from apps.telegram_bot.state import StateManager
from apps.telegram_bot.config import BotConfig
from apps.telegram_bot.formatter import (
    format_answer,
    format_citation_full,
    format_trace_full,
    format_retrieved_chunks,
    format_help_message,
    format_welcome_message,
    escape_markdown_v1,
)
from apps.telegram_bot.keyboards import (
    get_main_menu_keyboard,
    get_pipeline_keyboard,
    get_model_keyboard,
    get_corpus_keyboard,
    get_close_keyboard
)
from knowledge.models import Answer


logger = logging.getLogger(__name__)


class BotHandlers:
    """
    Bot handlers for all user interactions.
    
    Manages commands, text messages, and callback queries.
    """
    
    def __init__(
        self,
        state_manager: StateManager,
        pipeline_factory: Any  # Will create on demand
    ):
        """
        Initialize handlers.
        
        Args:
            state_manager: User state manager
            pipeline_factory: Factory function to create pipelines
        """
        self.state_manager = state_manager
        self.pipeline_factory = pipeline_factory
        
        # Cache for answers (for debug expand)
        self._answer_cache: Dict[str, Answer] = {}
    
    # ================== Command Handlers ==================
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        user = update.effective_user
        
        if not user:
            return
        
        # Initialize user state
        self.state_manager.get_state(user.id, user.username)
        
        logger.info(f"User {user.id} ({user.username}) started bot")
        
        await update.message.reply_text(
            format_welcome_message(),
            parse_mode="Markdown",
            reply_markup=get_main_menu_keyboard()
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        await update.message.reply_text(
            format_help_message(),
            parse_mode="Markdown"
        )
    
    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /settings command."""
        user = update.effective_user
        
        if not user:
            return
        
        state = self.state_manager.get_state(user.id)
        
        await update.message.reply_text(
            state.get_settings_summary(),
            parse_mode="Markdown",
            reply_markup=get_main_menu_keyboard()
        )
    
    async def pipeline_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /pipeline command."""
        user = update.effective_user
        
        if not user:
            return
        
        state = self.state_manager.get_state(user.id)
        
        await update.message.reply_text(
            "Select RAG pipeline:",
            reply_markup=get_pipeline_keyboard(state.pipeline)
        )
    
    async def model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /model command."""
        user = update.effective_user
        
        if not user:
            return
        
        state = self.state_manager.get_state(user.id)
        
        await update.message.reply_text(
            "Select LLM model:",
            reply_markup=get_model_keyboard(state.model)
        )
    
    async def corpus_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /corpus command."""
        user = update.effective_user
        
        if not user:
            return
        
        state = self.state_manager.get_state(user.id)
        
        await update.message.reply_text(
            "Select corpus profile:",
            reply_markup=get_corpus_keyboard(state.corpus)
        )
    
    async def debug_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /debug command - toggle debug mode."""
        user = update.effective_user
        
        if not user:
            return
        
        new_state = self.state_manager.toggle_debug(user.id)
        
        await update.message.reply_text(
            f"🐛 Debug mode: {'ON' if new_state else 'OFF'}",
            parse_mode="Markdown"
        )
    
    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /reset command - reset user settings."""
        user = update.effective_user
        
        if not user:
            return
        
        self.state_manager.reset_state(user.id)
        
        await update.message.reply_text(
            "✅ Settings reset to defaults",
            parse_mode="Markdown",
            reply_markup=get_main_menu_keyboard()
        )
    
    # ================== Message Handler ==================
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle user questions (text messages).
        
        Runs RAG pipeline and returns answer with citations.
        """
        user = update.effective_user
        message = update.message
        
        if not user or not message or not message.text:
            return
        
        query = message.text.strip()
        
        if not query:
            await message.reply_text("Please ask a question.")
            return
        
        # Get user state
        state = self.state_manager.get_state(user.id)
        
        logger.info(
            f"User {user.id} query: '{query[:50]}...' "
            f"[pipeline={state.pipeline}, model={state.model}]"
        )
        
        # Send "typing" action
        await context.bot.send_chat_action(
            chat_id=message.chat_id,
            action="typing"
        )
        
        try:
            # Create pipeline
            pipeline = self.pipeline_factory(
                pipeline_key=state.pipeline,
                model_key=state.model
            )
            
            # Run pipeline
            run_kwargs: Dict[str, Any] = {}
            # Map "top_k" from bot settings to pipeline-specific knobs.
            # (Most pipelines use **kwargs and have different parameter names.)
            if state.pipeline == "v1":
                run_kwargs["top_k"] = state.top_k
            elif state.pipeline == "v2":
                # v2 retrieves more, then reranks down.
                run_kwargs["retrieval_top_k"] = max(2 * state.top_k, 10)
                run_kwargs["rerank_top_k"] = state.top_k
            elif state.pipeline == "v3":
                # v3 uses multi-query fusion; keep fused set small to speed up generation.
                run_kwargs["retrieval_top_k"] = state.top_k
                run_kwargs["per_query_k"] = max(state.top_k, 5)
                # Bound LLM context size to avoid minute-long generations.
                run_kwargs["max_context_chunks"] = min(state.top_k, 6)
                run_kwargs["max_chunk_chars"] = 1200
                run_kwargs["max_tokens"] = 1024
            elif state.pipeline == "v4":
                run_kwargs["child_top_k"] = max(state.top_k, 5)
                run_kwargs["parent_top_k"] = min(5, max(3, state.top_k // 2))
            elif state.pipeline == "v5":
                run_kwargs["retrieval_top_k"] = max(2 * state.top_k, 10)
                run_kwargs["rerank_top_k"] = state.top_k

            answer = pipeline.run(
                query=query,
                corpus_profile=state.corpus,
                **run_kwargs,
            )
            
            # Update stats
            self.state_manager.increment_queries(user.id, query)
            
            # Cache answer for debug expansion
            answer_id = f"{user.id}_{state.queries_count}"
            self._answer_cache[answer_id] = answer
            
            # Format and send response
            response_text = format_answer(answer, show_debug=state.show_debug)
            
            try:
                await message.reply_text(
                    response_text,
                    parse_mode="Markdown"
                )
            except BadRequest as e:
                # Most common: broken Markdown entities due to unescaped model/citation text.
                logger.warning(f"Telegram send failed (Markdown), retrying as plain text: {e}")
                await message.reply_text(response_text)
            
            # If answer has citations, offer to view details
            if answer.has_citations() and not answer.is_refused():
                citation_info = (
                    f"\n💡 _Click citation numbers to view full quotes_"
                )
                # Note: In real implementation, we'd add inline buttons for each citation
                # For now, just inform user they can use debug mode
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            # Show a more helpful message for common setup issues
            if isinstance(e, FileNotFoundError):
                await message.reply_text(
                    "❌ Индексы для поиска не найдены или лежат не там.\n\n"
                    "Соберите индексы командой:\n"
                    "`python scripts/build_indices.py --profile public --strategy fixed`\n\n"
                    "Потом перезапустите бота.",
                    parse_mode="Markdown",
                )
                return
            await message.reply_text(
                "❌ Sorry, an error occurred while processing your question. "
                "Please try again or contact support.",
                parse_mode="Markdown"
            )
    
    # ================== Callback Handlers ==================
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle callback queries from inline buttons.
        
        Routes to specific handlers based on callback_data.
        """
        query = update.callback_query
        
        if not query or not query.data:
            return
        
        await query.answer()  # Acknowledge callback
        
        callback_data = query.data
        user = update.effective_user
        
        if not user:
            return
        
        logger.info(f"Callback from {user.id}: {callback_data}")
        
        # Route to appropriate handler
        if callback_data == "menu_main":
            await self._show_main_menu(query, user.id)
        
        elif callback_data == "menu_pipeline":
            await self._show_pipeline_menu(query, user.id)
        
        elif callback_data == "menu_model":
            await self._show_model_menu(query, user.id)
        
        elif callback_data == "menu_corpus":
            await self._show_corpus_menu(query, user.id)
        
        elif callback_data == "toggle_debug":
            await self._toggle_debug(query, user.id)
        
        elif callback_data == "show_settings":
            await self._show_settings(query, user.id)
        
        elif callback_data == "show_help":
            await self._show_help(query, user.id)
        
        elif callback_data.startswith("pipeline_"):
            await self._set_pipeline(query, user.id, callback_data)
        
        elif callback_data.startswith("model_"):
            await self._set_model(query, user.id, callback_data)
        
        elif callback_data.startswith("corpus_"):
            await self._set_corpus(query, user.id, callback_data)
        
        elif callback_data.startswith("cite_view_"):
            await self._view_citation(query, user.id, callback_data)
        
        elif callback_data.startswith("debug_"):
            await self._show_debug_info(query, user.id, callback_data)
        
        elif callback_data == "close":
            await query.message.delete()
    
    # ================== Callback Helpers ==================
    
    async def _show_main_menu(self, query: Any, user_id: int) -> None:
        """Show main menu."""
        await query.edit_message_text(
            "⚙️ *Main Menu*\n\nSelect an option:",
            parse_mode="Markdown",
            reply_markup=get_main_menu_keyboard()
        )
    
    async def _show_pipeline_menu(self, query: Any, user_id: int) -> None:
        """Show pipeline selection menu."""
        state = self.state_manager.get_state(user_id)
        
        await query.edit_message_text(
            "🔧 *Select Pipeline*\n\nChoose RAG pipeline variant:",
            parse_mode="Markdown",
            reply_markup=get_pipeline_keyboard(state.pipeline)
        )
    
    async def _show_model_menu(self, query: Any, user_id: int) -> None:
        """Show model selection menu."""
        state = self.state_manager.get_state(user_id)
        
        await query.edit_message_text(
            "🤖 *Select Model*\n\nChoose LLM model:",
            parse_mode="Markdown",
            reply_markup=get_model_keyboard(state.model)
        )
    
    async def _show_corpus_menu(self, query: Any, user_id: int) -> None:
        """Show corpus selection menu."""
        state = self.state_manager.get_state(user_id)
        
        await query.edit_message_text(
            "📚 *Select Corpus*\n\nChoose document corpus:",
            parse_mode="Markdown",
            reply_markup=get_corpus_keyboard(state.corpus)
        )
    
    async def _toggle_debug(self, query: Any, user_id: int) -> None:
        """Toggle debug mode."""
        new_state = self.state_manager.toggle_debug(user_id)
        
        await query.edit_message_text(
            f"🐛 Debug mode: **{'ON' if new_state else 'OFF'}**\n\n"
            f"{'Retrieval details will be shown with answers.' if new_state else 'Debug info hidden.'}",
            parse_mode="Markdown",
            reply_markup=get_main_menu_keyboard()
        )
    
    async def _show_settings(self, query: Any, user_id: int) -> None:
        """Show current settings."""
        state = self.state_manager.get_state(user_id)
        
        await query.edit_message_text(
            state.get_settings_summary(),
            parse_mode="Markdown",
            reply_markup=get_main_menu_keyboard()
        )
    
    async def _show_help(self, query: Any, user_id: int) -> None:
        """Show help message."""
        await query.edit_message_text(
            format_help_message(),
            parse_mode="Markdown",
            reply_markup=get_close_keyboard()
        )
    
    async def _set_pipeline(self, query: Any, user_id: int, callback_data: str) -> None:
        """Set pipeline preference."""
        pipeline_key = callback_data.replace("pipeline_", "")
        
        self.state_manager.update_pipeline(user_id, pipeline_key)
        
        pipeline_name = BotConfig.PIPELINES[pipeline_key]
        description = BotConfig.get_pipeline_description(pipeline_key)

        text_md = (
            f"✅ Pipeline set to: **{escape_markdown_v1(pipeline_name)}**\n\n"
            f"{escape_markdown_v1(description)}"
        )
        try:
            await query.edit_message_text(
                text_md,
                parse_mode="Markdown",
                reply_markup=get_main_menu_keyboard()
            )
        except BadRequest as e:
            logger.warning(f"Telegram edit failed (Markdown), retrying as plain text: {e}")
            await query.edit_message_text(
                f"✅ Pipeline set to: {pipeline_name}\n\n{description}",
                reply_markup=get_main_menu_keyboard()
            )
    
    async def _set_model(self, query: Any, user_id: int, callback_data: str) -> None:
        """Set model preference."""
        model_key = callback_data.replace("model_", "")
        
        self.state_manager.update_model(user_id, model_key)
        
        model_name = BotConfig.LLM_MODELS[model_key]
        description = BotConfig.get_model_description(model_key)

        text_md = (
            f"✅ Model set to: **{escape_markdown_v1(model_name)}**\n\n"
            f"{escape_markdown_v1(description)}"
        )
        try:
            await query.edit_message_text(
                text_md,
                parse_mode="Markdown",
                reply_markup=get_main_menu_keyboard()
            )
        except BadRequest as e:
            logger.warning(f"Telegram edit failed (Markdown), retrying as plain text: {e}")
            await query.edit_message_text(
                f"✅ Model set to: {model_name}\n\n{description}",
                reply_markup=get_main_menu_keyboard()
            )
    
    async def _set_corpus(self, query: Any, user_id: int, callback_data: str) -> None:
        """Set corpus preference."""
        corpus = callback_data.replace("corpus_", "")
        
        self.state_manager.update_corpus(user_id, corpus)

        text_md = f"✅ Corpus set to: **{escape_markdown_v1(corpus)}**"
        try:
            await query.edit_message_text(
                text_md,
                parse_mode="Markdown",
                reply_markup=get_main_menu_keyboard()
            )
        except BadRequest as e:
            logger.warning(f"Telegram edit failed (Markdown), retrying as plain text: {e}")
            await query.edit_message_text(
                f"✅ Corpus set to: {corpus}",
                reply_markup=get_main_menu_keyboard()
            )
    
    async def _view_citation(self, query: Any, user_id: int, callback_data: str) -> None:
        """View full citation details."""
        # Extract citation index from callback_data
        try:
            index = int(callback_data.replace("cite_view_", ""))
            
            # TODO: Retrieve citation from cache
            # For now, just acknowledge
            await query.edit_message_text(
                f"📄 Citation [{index}] details would be shown here.",
                parse_mode="Markdown",
                reply_markup=get_close_keyboard()
            )
        except (ValueError, KeyError):
            await query.edit_message_text(
                "❌ Citation not found.",
                reply_markup=get_close_keyboard()
            )
    
    async def _show_debug_info(self, query: Any, user_id: int, callback_data: str) -> None:
        """Show debug information."""
        # Parse callback data
        parts = callback_data.split("_")
        
        if len(parts) < 3:
            return
        
        debug_type = parts[1]  # chunks, timing, trace
        answer_id = "_".join(parts[2:])
        
        # Retrieve answer from cache
        answer = self._answer_cache.get(answer_id)
        
        if not answer or not answer.trace:
            await query.edit_message_text(
                "❌ Debug info not available.",
                reply_markup=get_close_keyboard()
            )
            return
        
        # Format based on type
        if debug_type == "trace":
            text = format_trace_full(answer.trace)
        elif debug_type == "timing":
            text = format_trace_full(answer.trace)  # Same for now
        else:
            text = "🐛 Debug info"
        
        await query.edit_message_text(
            text,
            parse_mode="Markdown",
            reply_markup=get_close_keyboard()
        )
    
    # ================== Error Handler ==================
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors."""
        logger.error("Exception while handling an update:", exc_info=context.error)
        
        # Notify user if possible
        if update and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "❌ An error occurred. Please try again later."
                )
            except Exception:
                pass

