# Telegram Bot MVP

Interactive Telegram bot for the QA Assistant, providing a user-friendly interface for querying the quantitative finance knowledge base.

## Features

### Core Functionality
- **Interactive Chat**: Ask questions in natural language
- **Citations Display**: View source documents and page references
- **Multiple RAG Pipelines**: Choose between v1-v5 implementations
- **LLM Model Selection**: Switch between DeepSeek models
- **Debug Mode**: View retrieval details and timing information

### User Interface
- **Command Menu**: Access all features via commands
- **Inline Keyboards**: Interactive buttons for settings
- **Settings Management**: Persistent per-user preferences
- **Help System**: Built-in documentation

## Setup

### 1. Get Telegram Bot Token

1. Open Telegram and search for `@BotFather`
2. Start a chat and send `/newbot`
3. Follow the prompts to create your bot
4. Copy the bot token provided

### 2. Configure Environment

Create a `.env` file (or copy from `env.example`):

```bash
# DeepSeek API
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather

# Optional: Custom data directory
# DATA_DIR=./data
```

### 3. Ensure Indices Are Built

Before running the bot, make sure you have built the indices:

```bash
# Parse corpus
python scripts/parse_corpus.py

# Build indices
python scripts/build_indices.py
```

This will create:
- `data/indices/public/vector.index` - Vector embeddings index
- `data/indices/public/bm25.index` - BM25 text index
- `data/indices/public/chunks.db` - Chunk metadata store

### 4. Run the Bot

```bash
python scripts/run_telegram_bot.py
```

Or directly:

```bash
python -m apps.telegram_bot.bot
```

The bot will start polling and log to console and `telegram_bot.log`.

## Usage

### Commands

- `/start` - Start the bot and show welcome message
- `/help` - Show help and available commands
- `/settings` - View current settings
- `/pipeline` - Select RAG pipeline (v1-v5)
- `/model` - Select LLM model
- `/corpus` - Select corpus profile
- `/debug` - Toggle debug mode
- `/reset` - Reset settings to defaults

### Asking Questions

Just type your question in natural language:

```
What is the Black-Scholes formula for call options?
```

The bot will:
1. Retrieve relevant chunks from the corpus
2. Generate an answer with citations
3. Display page references and quotes
4. Show debug info (if enabled)

### Selecting Pipeline

Use `/pipeline` or the menu to choose:

- **v1** - Dense retrieval (vector search only)
- **v2** - Hybrid (BM25 + vectors) with reranking
- **v3** - Multi-query expansion with fusion
- **v4** - Parent-child retrieval (context expansion)
- **v5** - Evidence validation (claim verification)

### Debug Mode

Enable debug mode with `/debug` to see:
- Number of retrieved/reranked/final chunks
- Query expansions (for multi-query pipelines)
- Timing information (retrieval, generation, total)
- Pipeline metadata

## Architecture

```
apps/telegram_bot/
├── __init__.py           # Package init
├── bot.py                # Main bot application
├── config.py             # Configuration and constants
├── state.py              # User state management
├── handlers.py           # Command and message handlers
├── keyboards.py          # Inline keyboard layouts
├── formatter.py          # Message formatting utilities
├── pipeline_factory.py   # RAG pipeline factory
└── README.md            # This file
```

### Component Responsibilities

- **bot.py**: Application setup, handler registration, lifecycle
- **config.py**: Bot settings, pipeline/model definitions, validation
- **state.py**: Per-user state tracking (pipeline, model, debug mode)
- **handlers.py**: Command processing, message handling, callbacks
- **keyboards.py**: Inline button layouts for interactive menus
- **formatter.py**: Answer/citation/debug formatting for Telegram
- **pipeline_factory.py**: Creates RAG pipeline instances on demand

### Flow

```
User Message
    ↓
handlers.handle_message()
    ↓
state_manager.get_state() → Get user preferences
    ↓
pipeline_factory.create_pipeline() → Create RAG pipeline
    ↓
pipeline.run(query) → Execute RAG
    ↓
formatter.format_answer() → Format response
    ↓
Send to Telegram
```

## Development

### Adding New Commands

1. Add handler method in `handlers.py`:
```python
async def my_command(self, update, context):
    await update.message.reply_text("Response")
```

2. Register in `bot.py`:
```python
app.add_handler(CommandHandler("mycommand", h.my_command))
```

### Adding New Keyboards

Define keyboard layout in `keyboards.py`:
```python
def get_my_keyboard() -> InlineKeyboardMarkup:
    keyboard = [[InlineKeyboardButton("Label", callback_data="action")]]
    return InlineKeyboardMarkup(keyboard)
```

Handle callbacks in `handlers.py`:
```python
async def handle_callback(self, update, context):
    if query.data == "action":
        # Handle action
```

### Testing

Test bot locally:
```bash
# With mock updates (TODO: add test script)
python tests/test_telegram_bot.py
```

Test with real Telegram:
1. Run bot: `python scripts/run_telegram_bot.py`
2. Open Telegram, search for your bot
3. Send `/start`

## Troubleshooting

### "Configuration error: TELEGRAM_BOT_TOKEN not set"
- Make sure `.env` file exists with `TELEGRAM_BOT_TOKEN`
- Or export: `export TELEGRAM_BOT_TOKEN=your_token`

### "Vector index not found"
- Run `python scripts/build_indices.py` first
- Make sure `data/indices/public/` directory exists

### "Module not found" errors
- Run from project root: `python scripts/run_telegram_bot.py`
- Or set PYTHONPATH: `export PYTHONPATH=.`

### Bot not responding
- Check logs in `telegram_bot.log`
- Verify token is correct
- Test connection: `curl https://api.telegram.org/bot<TOKEN>/getMe`

## Future Enhancements

- [ ] Persistent user state (database)
- [ ] Citation expansion inline buttons
- [ ] Query history and favorites
- [ ] Multi-user analytics dashboard
- [ ] Rate limiting and quota management
- [ ] Admin commands and monitoring
- [ ] Webhook mode (instead of polling)
- [ ] Interactive citation viewer with context
- [ ] Export conversation history
- [ ] Multi-language support

## References

- [Telegram Bot API](https://core.telegram.org/bots/api)
- [python-telegram-bot](https://docs.python-telegram-bot.org/)
- [DeepSeek API](https://platform.deepseek.com/docs)

