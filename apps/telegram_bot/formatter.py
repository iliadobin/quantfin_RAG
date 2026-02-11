"""
Formatters for answers, citations, and debug information.

Formats RAG pipeline outputs for Telegram display with proper escaping
and length limits.
"""
from typing import List, Optional
from knowledge.models import Answer, Citation, RetrievalTrace, RetrievedChunk
from apps.telegram_bot.config import BotConfig


def escape_markdown(text: str) -> str:
    """
    Escape special characters for Telegram MarkdownV2.
    
    Args:
        text: Text to escape
        
    Returns:
        Escaped text
    """
    # Characters that need escaping in MarkdownV2
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    
    return text


def escape_markdown_v1(text: str) -> str:
    """
    Escape special characters for Telegram legacy Markdown (parse_mode="Markdown").

    Note: This is NOT MarkdownV2 escaping.
    We only escape characters that commonly break Telegram Markdown entities.
    """
    if text is None:
        return ""
    # Backslash first
    text = text.replace("\\", "\\\\")
    for char in ("_", "*", "`", "[", "]", "(", ")"):
        text = text.replace(char, f"\\{char}")
    return text


def truncate_text(text: str, max_length: int = BotConfig.MAX_MESSAGE_LENGTH) -> str:
    """
    Truncate text to fit Telegram message limits.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - 50] + "\n\n... (message truncated)"


def format_answer(answer: Answer, show_debug: bool = False) -> str:
    """
    Format answer for display.
    
    Args:
        answer: Answer object from RAG pipeline
        show_debug: Whether to include debug information
        
    Returns:
        Formatted message text (with Markdown)
    """
    if answer.is_refused():
        # Refusal message
        msg = "🚫 *Unable to Answer*\n\n"
        msg += f"{escape_markdown_v1(answer.text)}\n\n"
        if answer.refusal_reason:
            msg += f"_Reason: {escape_markdown_v1(answer.refusal_reason)}_"
        return msg
    
    # Main answer
    msg = "💡 *Answer*\n\n"
    msg += f"{escape_markdown_v1(answer.text)}\n\n"
    
    # Citations
    if answer.has_citations():
        msg += "📚 *References*\n\n"
        for i, citation in enumerate(answer.citations, 1):
            msg += format_citation_short(citation, index=i)
            msg += "\n"
    else:
        msg += "_⚠️ No citations provided_\n"
    
    # Confidence
    confidence_emoji = "✅" if answer.confidence >= 0.8 else "⚠️" if answer.confidence >= 0.5 else "❌"
    msg += f"\n{confidence_emoji} Confidence: `{answer.confidence:.2f}`"
    
    # Debug info (if enabled)
    if show_debug and answer.trace:
        msg += "\n\n" + format_trace_summary(answer.trace)
    
    return truncate_text(msg)


def format_citation_short(citation: Citation, index: int) -> str:
    """
    Format citation in compact form.
    
    Args:
        citation: Citation object
        index: Citation number (1-indexed)
        
    Returns:
        Formatted citation string
    """
    ref = escape_markdown_v1(citation.format_page_reference())
    quote = citation.quote or ""
    quote_preview = quote[:80] + "..." if len(quote) > 80 else quote
    quote_preview = escape_markdown_v1(quote_preview)
    
    return f"[{index}] {ref}\n    \"{quote_preview}\""


def format_citation_full(citation: Citation, index: int) -> str:
    """
    Format citation with full details.
    
    Args:
        citation: Citation object
        index: Citation number (1-indexed)
        
    Returns:
        Formatted detailed citation
    """
    msg = f"📄 *Citation [{index}]*\n\n"
    msg += f"*Document:* {escape_markdown_v1(citation.doc_id)}\n"
    
    # Page span
    if citation.page_span.start_page == citation.page_span.end_page:
        msg += f"*Page:* {citation.page_span.start_page}\n"
    else:
        msg += f"*Pages:* {citation.page_span.start_page}–{citation.page_span.end_page}\n"
    
    msg += f"*Relevance Score:* {citation.score:.3f}\n"
    msg += f"*Retriever:* {escape_markdown_v1(citation.retriever_tag)}\n\n"
    
    msg += f"*Quote:*\n{escape_markdown_v1(citation.quote or '')}\n"
    
    return truncate_text(msg)


def format_trace_summary(trace: RetrievalTrace) -> str:
    """
    Format retrieval trace summary.
    
    Args:
        trace: Retrieval trace object
        
    Returns:
        Formatted trace summary
    """
    msg = "🐛 *Debug Info*\n"
    msg += f"• Retrieved chunks: {trace.retrieved_chunks_count}\n"
    
    if trace.reranked_chunks_count > 0:
        msg += f"• Reranked chunks: {trace.reranked_chunks_count}\n"
    
    msg += f"• Final chunks: {trace.final_chunks_count}\n"
    
    if trace.expanded_queries:
        msg += f"• Query expansions: {len(trace.expanded_queries)}\n"
    
    if trace.retrieval_time_ms:
        msg += f"• Retrieval time: `{trace.retrieval_time_ms:.1f}ms`\n"
    
    if trace.generation_time_ms:
        msg += f"• Generation time: `{trace.generation_time_ms:.1f}ms`\n"
    
    if "total_time_ms" in trace.metadata:
        msg += f"• Total time: `{trace.metadata['total_time_ms']:.1f}ms`\n"
    
    return msg


def format_trace_full(trace: RetrievalTrace) -> str:
    """
    Format full retrieval trace with all details.
    
    Args:
        trace: Retrieval trace object
        
    Returns:
        Formatted full trace
    """
    msg = "🔍 *Full Retrieval Trace*\n\n"
    
    msg += f"*Original Query:*\n{escape_markdown_v1(trace.query)}\n\n"
    
    # Expanded queries
    if trace.expanded_queries:
        msg += f"*Expanded Queries ({len(trace.expanded_queries)}):*\n"
        for i, q in enumerate(trace.expanded_queries, 1):
            preview = q[:100] + ("..." if len(q) > 100 else "")
            msg += f"{i}. {escape_markdown_v1(preview)}\n"
        msg += "\n"
    
    # Statistics
    msg += "*Statistics:*\n"
    msg += f"• Retrieved: {trace.retrieved_chunks_count} chunks\n"
    
    if trace.reranked_chunks_count > 0:
        msg += f"• Reranked: {trace.reranked_chunks_count} chunks\n"
    
    msg += f"• Final: {trace.final_chunks_count} chunks\n\n"
    
    # Timing
    msg += "*Timing:*\n"
    if trace.retrieval_time_ms:
        msg += f"• Retrieval: {trace.retrieval_time_ms:.1f}ms\n"
    if trace.generation_time_ms:
        msg += f"• Generation: {trace.generation_time_ms:.1f}ms\n"
    if "total_time_ms" in trace.metadata:
        msg += f"• Total: {trace.metadata['total_time_ms']:.1f}ms\n"
    
    # Metadata
    if trace.metadata:
        msg += "\n*Metadata:*\n"
        for key, value in trace.metadata.items():
            if key != "total_time_ms":  # Already shown above
                msg += f"• {escape_markdown_v1(str(key))}: {escape_markdown_v1(str(value))}\n"
    
    return truncate_text(msg)


def format_retrieved_chunks(chunks: List[RetrievedChunk], limit: int = 5) -> str:
    """
    Format retrieved chunks for debug display.
    
    Args:
        chunks: List of retrieved chunks
        limit: Maximum number of chunks to display
        
    Returns:
        Formatted chunks list
    """
    msg = f"📦 *Retrieved Chunks* (showing top {min(limit, len(chunks))} of {len(chunks)})\n\n"
    
    for i, retrieved in enumerate(chunks[:limit], 1):
        chunk = retrieved.chunk
        msg += f"*[{i}]* Score: `{retrieved.score:.3f}` | Tag: `{retrieved.retriever_tag}`\n"
        msg += f"Doc: {escape_markdown_v1(chunk.doc_id)} | Pages: {chunk.page_span.start_page}–{chunk.page_span.end_page}\n"
        
        # Preview text
        preview = chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
        # Avoid code fences (```), as chunk text can contain backticks and break Markdown parsing.
        msg += f"{escape_markdown_v1(preview)}\n"
    
    if len(chunks) > limit:
        msg += f"\n_... and {len(chunks) - limit} more chunks_"
    
    return truncate_text(msg)


def format_help_message() -> str:
    """Format help message."""
    msg = """
📖 *QA Assistant Help*

*Available Commands:*
/start - Start the bot and show main menu
/settings - View current settings
/pipeline - Select RAG pipeline (v1-v5)
/model - Select LLM model
/corpus - Select corpus profile
/debug - Toggle debug mode
/reset - Reset settings to defaults
/help - Show this help message

*How to Use:*
1️⃣ Configure your preferences using commands or menu buttons
2️⃣ Ask questions about quantitative finance topics
3️⃣ Review answers with supporting citations
4️⃣ Use debug mode to see retrieval details

*Pipeline Options:*
• v1: Dense retrieval (vector search)
• v2: Hybrid (BM25 + vectors + rerank)
• v3: Multi-query expansion
• v4: Parent-child retrieval
• v5: Evidence validation

*Tips:*
💡 Be specific in your questions
💡 Use technical terms when applicable
💡 Enable debug mode to understand retrieval
💡 Check citations for evidence
"""
    return msg.strip()


def format_welcome_message() -> str:
    """Format welcome message."""
    msg = """
👋 *Welcome to QA Assistant!*

I'm your quantitative finance research assistant. I can answer questions about derivatives pricing, risk management, and related topics using a curated corpus of academic papers and regulatory documents.

🎯 *What I can do:*
• Answer technical questions with citations
• Retrieve relevant information from documents
• Show evidence and page references
• Provide multiple retrieval strategies

⚙️ *Get Started:*
Use /settings to configure your preferences, or just start asking questions!

Type /help for more information.
"""
    return msg.strip()

