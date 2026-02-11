#!/usr/bin/env python3
"""
Script to run the Telegram bot.

Usage:
    python scripts/run_telegram_bot.py

Environment variables:
    TELEGRAM_BOT_TOKEN: Bot token from @BotFather
    DEEPSEEK_API_KEY: DeepSeek API key
    DATA_DIR: Path to data directory (default: ./data)
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from apps.telegram_bot.bot import main


if __name__ == "__main__":
    main()

