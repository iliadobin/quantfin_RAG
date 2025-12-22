"""
LLM-as-a-judge for answer quality evaluation.

Uses DeepSeek API with cache-friendly prompts to evaluate:
- Answer correctness
- Answer completeness
- Relevance to question
- Overall quality score

Optimized for minimal token usage through:
- Batching multiple evaluations
- Cache-friendly prompt structure
- Deterministic prompts for cache hits
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
import json


class JudgeScore(BaseModel):
    """Score from LLM judge for a single answer."""
    
    # Scores [0-5]
    correctness: int = Field(ge=0, le=5, description="Factual correctness")
    completeness: int = Field(ge=0, le=5, description="Answer completeness")
    relevance: int = Field(ge=0, le=5, description="Relevance to question")
    
    # Overall
    overall_score: float = Field(ge=0.0, le=5.0)
    
    # Explanation
    reasoning: str = ""
    
    # Metadata
    example_id: str


class JudgeBatchResult(BaseModel):
    """Results from batch judging."""
    scores: List[JudgeScore]
    
    # Aggregate statistics
    mean_correctness: float = 0.0
    mean_completeness: float = 0.0
    mean_relevance: float = 0.0
    mean_overall: float = 0.0
    
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


class LLMJudge:
    """
    LLM-as-a-judge evaluator.
    
    Uses cache-friendly prompts to minimize token costs.
    """
    
    def __init__(self, llm_client, model: str = "deepseek-chat"):
        """
        Initialize judge.
        
        Args:
            llm_client: DeepSeek client instance
            model: Model name to use
        """
        self.llm = llm_client
        self.model = model
        
        # Cache-friendly system prompt (stays constant)
        self.system_prompt = """You are an expert evaluator for question-answering systems in quantitative finance.

Your task is to evaluate answers to questions about derivatives, pricing, risk management, and related topics.

For each answer, provide scores (0-5) for:
1. **Correctness**: Is the answer factually correct based on the gold standard?
2. **Completeness**: Does it cover all important aspects of the question?
3. **Relevance**: Does it directly address what was asked?

Also provide:
- **Overall Score**: Average of the three scores
- **Reasoning**: Brief explanation of your scores (1-2 sentences)

Output format (JSON):
{
  "correctness": <0-5>,
  "completeness": <0-5>,
  "relevance": <0-5>,
  "overall_score": <float>,
  "reasoning": "<explanation>"
}

Scoring rubric:
- 5: Excellent - fully correct, complete, and relevant
- 4: Good - mostly correct with minor issues
- 3: Acceptable - correct but incomplete or somewhat off-topic
- 2: Poor - significant errors or missing key information
- 1: Very poor - mostly incorrect or irrelevant
- 0: Completely wrong or refuses to answer when should answer
"""
    
    def _format_evaluation_prompt(
        self,
        question: str,
        predicted_answer: str,
        gold_answer: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Format single evaluation prompt.
        
        Args:
            question: The question
            predicted_answer: System's answer
            gold_answer: Gold standard answer (if available)
            context: Additional context
            
        Returns:
            Formatted prompt
        """
        prompt_parts = [
            f"**Question**: {question}",
            f"\n**Predicted Answer**: {predicted_answer}"
        ]
        
        if gold_answer:
            prompt_parts.append(f"\n**Gold Standard Answer**: {gold_answer}")
        
        if context:
            prompt_parts.append(f"\n**Additional Context**: {context}")
        
        prompt_parts.append("\n**Evaluation** (JSON format):")
        
        return "\n".join(prompt_parts)
    
    def evaluate_single(
        self,
        question: str,
        predicted_answer: str,
        gold_answer: Optional[str] = None,
        example_id: str = "unknown"
    ) -> JudgeScore:
        """
        Evaluate a single answer.
        
        Args:
            question: The question
            predicted_answer: System's answer
            gold_answer: Gold standard (optional)
            example_id: Example identifier
            
        Returns:
            JudgeScore with evaluation
        """
        user_prompt = self._format_evaluation_prompt(
            question, predicted_answer, gold_answer
        )
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.llm.chat(
            messages=messages,
            model=self.model,
            temperature=0.0,  # Deterministic for reproducibility
            response_format={"type": "json_object"}
        )
        
        # Parse response
        try:
            result = json.loads(response['content'])
            score = JudgeScore(
                correctness=result.get('correctness', 0),
                completeness=result.get('completeness', 0),
                relevance=result.get('relevance', 0),
                overall_score=result.get('overall_score', 0.0),
                reasoning=result.get('reasoning', ''),
                example_id=example_id
            )
        except Exception as e:
            # Fallback on parse error
            score = JudgeScore(
                correctness=0,
                completeness=0,
                relevance=0,
                overall_score=0.0,
                reasoning=f"Parse error: {str(e)}",
                example_id=example_id
            )
        
        return score
    
    def evaluate_batch(
        self,
        evaluations: List[Dict[str, Any]],
        batch_size: int = 5
    ) -> JudgeBatchResult:
        """
        Evaluate multiple answers in batches for efficiency.
        
        Args:
            evaluations: List of dicts with:
                - example_id: str
                - question: str
                - predicted_answer: str
                - gold_answer: str (optional)
            batch_size: Number of examples per batch (for batching in prompt)
            
        Returns:
            JudgeBatchResult with all scores
        """
        all_scores = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        
        # Process in batches
        for i in range(0, len(evaluations), batch_size):
            batch = evaluations[i:i + batch_size]
            
            # For simplicity, evaluate one by one
            # Could optimize further by combining multiple in one prompt
            for item in batch:
                score = self.evaluate_single(
                    question=item['question'],
                    predicted_answer=item['predicted_answer'],
                    gold_answer=item.get('gold_answer'),
                    example_id=item['example_id']
                )
                all_scores.append(score)
                
                # Note: would track token usage from response
                # For now, estimate
                total_prompt_tokens += 500  # Rough estimate
                total_completion_tokens += 100
        
        # Compute aggregate statistics
        if all_scores:
            mean_correctness = sum(s.correctness for s in all_scores) / len(all_scores)
            mean_completeness = sum(s.completeness for s in all_scores) / len(all_scores)
            mean_relevance = sum(s.relevance for s in all_scores) / len(all_scores)
            mean_overall = sum(s.overall_score for s in all_scores) / len(all_scores)
        else:
            mean_correctness = mean_completeness = mean_relevance = mean_overall = 0.0
        
        result = JudgeBatchResult(
            scores=all_scores,
            mean_correctness=mean_correctness,
            mean_completeness=mean_completeness,
            mean_relevance=mean_relevance,
            mean_overall=mean_overall,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            cached_tokens=total_cached_tokens
        )
        
        return result


def judge_answer_quality(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    llm_client,
    model: str = "deepseek-chat",
    batch_size: int = 5
) -> JudgeBatchResult:
    """
    Convenience function to judge answer quality.
    
    Args:
        predictions: List of predictions with example_id, answer
        ground_truth: List of ground truth with example_id, gold_answer
        llm_client: DeepSeek client
        model: Model to use
        batch_size: Batch size for evaluation
        
    Returns:
        JudgeBatchResult with all scores
    """
    judge = LLMJudge(llm_client, model)
    
    # Build mapping
    gt_map = {item['example_id']: item.get('gold_answer', '') for item in ground_truth}
    
    # Prepare evaluations
    evaluations = []
    for pred in predictions:
        evaluations.append({
            'example_id': pred['example_id'],
            'question': pred.get('question', ''),
            'predicted_answer': pred['answer'].text,
            'gold_answer': gt_map.get(pred['example_id'], '')
        })
    
    return judge.evaluate_batch(evaluations, batch_size)

