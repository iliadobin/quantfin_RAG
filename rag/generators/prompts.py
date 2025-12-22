"""
Prompt templates for answer generation.

Designed for cache-friendly structure with DeepSeek API.
"""

# System prompt - kept consistent for cache hits
SYSTEM_PROMPT = """You are an expert assistant for quantitative finance and derivatives.

Your role is to answer questions based ONLY on the provided context documents.

Requirements:
1. Answer must be grounded in the provided context
2. Include inline citations [1], [2], etc. for claims
3. If context is insufficient, say "I cannot answer this question based on the provided documents"
4. Be precise and technical - this is for finance professionals
5. Include relevant formulas, assumptions, and conditions when applicable

Do not use external knowledge. Only use information from the context provided."""


def format_context(chunks, max_chunks: int = 10) -> str:
    """
    Format retrieved chunks as context.
    
    Args:
        chunks: List of RetrievedChunk objects
        max_chunks: Maximum chunks to include
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, retrieved in enumerate(chunks[:max_chunks], start=1):
        chunk = retrieved.chunk
        doc_id = chunk.doc_id
        page_info = f"pp. {chunk.page_span.start_page}-{chunk.page_span.end_page}"
        
        context_parts.append(
            f"[{i}] Source: {doc_id}, {page_info}\n{chunk.text}\n"
        )
    
    return "\n".join(context_parts)


def build_qa_prompt(query: str, chunks, max_chunks: int = 10) -> str:
    """
    Build user prompt for QA with citations.
    
    Args:
        query: User question
        chunks: Retrieved chunks
        max_chunks: Maximum chunks to include
        
    Returns:
        Formatted user prompt
    """
    context = format_context(chunks, max_chunks)
    
    return f"""Context documents:

{context}

Question: {query}

Instructions:
- Answer the question using ONLY the context above
- Include citations [1], [2], etc. when making claims
- If the context doesn't contain the answer, say so explicitly
- Be concise but complete

Answer:"""


# Alternative prompt for structured extraction
SYSTEM_PROMPT_STRUCTURED = """You are an expert assistant for quantitative finance that extracts structured information.

Your role is to extract information from context documents and return it in the requested format.

Requirements:
1. Extract information ONLY from provided context
2. Follow the output schema exactly
3. If information is missing, use null or indicate explicitly
4. Be precise - this is for financial analysis

Do not use external knowledge. Only use information from the context provided."""


def build_structured_prompt(query: str, chunks, schema: dict, max_chunks: int = 10) -> str:
    """
    Build prompt for structured extraction.
    
    Args:
        query: Extraction task description
        chunks: Retrieved chunks
        schema: JSON schema for output
        max_chunks: Maximum chunks to include
        
    Returns:
        Formatted prompt
    """
    context = format_context(chunks, max_chunks)
    schema_str = str(schema)
    
    return f"""Context documents:

{context}

Task: {query}

Output schema:
{schema_str}

Extract the requested information and format as JSON following the schema.
If information is not found in the context, use null values.

JSON output:"""


# Prompt for evidence validation
SYSTEM_PROMPT_VALIDATION = """You are a validation assistant for quantitative finance Q&A.

Your role is to verify that answer claims are supported by context evidence.

For each claim in the answer, determine:
1. Is it supported by the provided context? (yes/no)
2. Which context chunks support it? (chunk numbers)
3. Confidence level (high/medium/low)

Be strict - only mark as supported if the context explicitly contains the information."""


def build_validation_prompt(answer: str, chunks, max_chunks: int = 10) -> str:
    """
    Build prompt for evidence validation.
    
    Args:
        answer: Generated answer to validate
        chunks: Context chunks used
        max_chunks: Maximum chunks
        
    Returns:
        Validation prompt
    """
    context = format_context(chunks, max_chunks)
    
    return f"""Context:

{context}

Answer to validate:
{answer}

For each factual claim in the answer, verify:
1. Is it supported by the context?
2. Which chunks [1], [2], etc. support it?
3. Confidence: high/medium/low

Output format:
Claim: <claim text>
Supported: yes/no
Sources: [chunk numbers]
Confidence: high/medium/low

Analysis:"""

