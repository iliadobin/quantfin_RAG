#!/usr/bin/env python3
"""
Generate "production" benchmark datasets (DS1–DS5) filled with REAL examples.

Targets:
- doc_id must exist in `configs/corpus_public.yaml`
- DS2 qrels must use REAL chunk_id from `data/indices/public/<strategy>/chunks.jsonl`
- DS1/DS4/DS5 citations should reference real doc_id/page_span and include a real quote

This script is intentionally conservative: it uses a small, curated set of examples built from
the locally ingested public corpus, without any external LLM calls.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.datasets.generator import DatasetGenerator  # noqa: E402
from benchmarks.datasets.loader import save_dataset  # noqa: E402
from benchmarks.datasets.validator import DatasetValidator  # noqa: E402
from benchmarks.schemas import (  # noqa: E402
    DS1Example,
    DS2Example,
    DS2Qrel,
    DS3Example,
    DS4Example,
    DS4HopInfo,
    DS5Example,
    DS5StructuredOutput,
)
from knowledge.models import Citation, PageSpan  # noqa: E402


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _snippet_around(text: str, anchor: Optional[str], max_len: int = 320) -> str:
    t = _norm_ws(text)
    if not anchor:
        return t[:max_len]
    low = t.lower()
    a = anchor.lower()
    idx = low.find(a)
    if idx < 0:
        return t[:max_len]
    half = max_len // 2
    start = max(0, idx - half)
    end = min(len(t), idx + len(a) + half)
    return t[start:end].strip()


def _first_sentence(text: str, min_len: int = 40, max_len: int = 280) -> str:
    t = _norm_ws(text)
    # Split on sentence-ish boundaries; keep it simple/robust for PDF text.
    parts = re.split(r"(?<=[.!?])\s+", t)
    acc = ""
    for p in parts:
        if not p:
            continue
        if acc:
            cand = f"{acc} {p}"
        else:
            cand = p
        if len(cand) >= min_len:
            return cand[:max_len].strip()
        acc = cand
    return t[:max_len].strip()


def _answer_from_anchor(text: str, anchor: Optional[str], max_len: int = 320) -> str:
    """
    Produce a human-readable gold answer by starting near an anchor phrase
    (avoids PDF headers like 'Chapter 1' / page numbers).
    """
    if not anchor:
        return _first_sentence(text, max_len=max_len)
    t = _norm_ws(text)
    low = t.lower()
    a = anchor.lower()
    idx = low.find(a)
    if idx < 0:
        return _first_sentence(t, max_len=max_len)
    return _first_sentence(t[idx:], max_len=max_len)


def _load_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def _find_chunk(
    chunks: Sequence[Dict[str, Any]],
    *,
    doc_id: Optional[str] = None,
    contains_all: Optional[Sequence[str]] = None,
    contains_any: Optional[Sequence[str]] = None,
    regex: Optional[str] = None,
) -> Dict[str, Any]:
    contains_all = contains_all or []
    contains_any = contains_any or []
    rx = re.compile(regex, re.IGNORECASE) if regex else None

    for ch in chunks:
        if doc_id and ch.get("doc_id") != doc_id:
            continue
        text = ch.get("text", "")
        low = text.lower()

        if any(term.lower() not in low for term in contains_all):
            continue
        if contains_any and not any(term.lower() in low for term in contains_any):
            continue
        if rx and not rx.search(text):
            continue
        return ch

    criteria = {
        "doc_id": doc_id,
        "contains_all": list(contains_all),
        "contains_any": list(contains_any),
        "regex": regex,
    }
    raise RuntimeError(f"Could not find a chunk matching criteria: {criteria}")


def _citation_from_chunk(ch: Dict[str, Any], *, anchor: Optional[str] = None) -> Citation:
    ps = ch["page_span"]
    return Citation(
        doc_id=ch["doc_id"],
        page_span=PageSpan(
            start_page=int(ps["start_page"]),
            end_page=int(ps["end_page"]),
            start_char=ps.get("start_char"),
            end_char=ps.get("end_char"),
        ),
        quote=_snippet_around(ch["text"], anchor=anchor, max_len=360),
        score=1.0,
        retriever_tag="gold",
        chunk_id=ch["id"],
    )


@dataclass(frozen=True)
class ChunkRef:
    doc_id: str
    contains_all: Sequence[str] = ()
    contains_any: Sequence[str] = ()
    regex: Optional[str] = None
    anchor: Optional[str] = None


def _require_chunk(chunks: Sequence[Dict[str, Any]], ref: ChunkRef) -> Dict[str, Any]:
    return _find_chunk(
        chunks,
        doc_id=ref.doc_id,
        contains_all=ref.contains_all,
        contains_any=ref.contains_any,
        regex=ref.regex,
    )


def build_production_datasets(index_strategy: str = "fixed") -> None:
    # Paths
    chunks_path = project_root / "data" / "indices" / "public" / index_strategy / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk store not found: {chunks_path}. Build indices first: "
            f"`python scripts/build_indices.py --strategy {index_strategy}`."
        )

    out_dir = project_root / "benchmarks" / "datasets" / "production"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = _load_chunks(chunks_path)
    gen = DatasetGenerator(corpus_profile="public")
    validator = DatasetValidator(strict=False)

    # ---------------------------------------------------------------------
    # DS1 (Factual QA)
    # ---------------------------------------------------------------------
    ds1 = gen.create_empty_ds1(description="Production DS1 for public corpus (real citations)")

    # Curated anchors we can reliably locate in the ingested public corpus.
    ch_risk_def = _require_chunk(
        chunks,
        ChunkRef(
            doc_id="riskmetrics_1996",
            contains_all=("We define risk as the degree of uncertainty",),
            anchor="We define risk as the degree of uncertainty",
        ),
    )
    ch_var_def = _require_chunk(
        chunks,
        ChunkRef(
            doc_id="riskmetrics_1996",
            contains_all=("Value-at-Risk is a measure of the maximum potential change in value",),
            anchor="Value-at-Risk is a measure",
        ),
    )
    ch_rm_components = _require_chunk(
        chunks,
        ChunkRef(
            doc_id="riskmetrics_1996",
            contains_all=("RiskMetrics has", "three basic components"),
            anchor="three basic components",
        ),
    )
    ch_delta_def = _require_chunk(
        chunks,
        ChunkRef(
            doc_id="riskmetrics_1996",
            contains_all=("Delta is the first derivative of the option price",),
            anchor="Delta is the first derivative",
        ),
    )
    ch_mc_steps = _require_chunk(
        chunks,
        ChunkRef(
            doc_id="riskmetrics_1996",
            contains_all=("The Monte Carlo methodology consists of three major steps",),
            anchor="three major steps",
        ),
    )
    ch_es_min_standards = _require_chunk(
        chunks,
        ChunkRef(
            doc_id="bcbs_frtb_2019",
            contains_all=("Calculation of expected shortfall", "97.5th percentile"),
            anchor="97.5th percentile",
        ),
    )
    ch_gbm_def = _require_chunk(
        chunks,
        ChunkRef(
            doc_id="arxiv_2510_27277v1",
            contains_all=("A Geometric Brownian Motion", "logarithm"),
            anchor="Geometric Brownian Motion",
        ),
    )
    ch_ito_def = _require_chunk(
        chunks,
        ChunkRef(
            doc_id="arxiv_2510_27277v1",
            contains_all=("Ito’s lemma is an identity",),
            anchor="Ito’s lemma is an identity",
        ),
    )
    ch_bs_assumptions = _require_chunk(
        chunks,
        ChunkRef(
            doc_id="arxiv_2510_27277v1",
            contains_all=("The risk-free interest rate is constant",),
            anchor="The risk-free interest rate is constant",
        ),
    )
    ch_fd_methods = _require_chunk(
        chunks,
        ChunkRef(
            doc_id="arxiv_2408_15416v1",
            contains_all=("finite difference methods", "parabolic"),
            anchor="finite difference methods",
        ),
    )

    ds1_examples: List[DS1Example] = [
        DS1Example(
            id="ds1_pub_001",
            question="How does RiskMetrics define risk?",
            gold_answer=_answer_from_anchor(ch_risk_def["text"], "We define risk", max_len=320),
            gold_citations=[_citation_from_chunk(ch_risk_def, anchor="We define risk")],
            topic="risk_definition",
            difficulty="easy",
        ),
        DS1Example(
            id="ds1_pub_002",
            question="What is Value-at-Risk (VaR) according to the RiskMetrics technical document?",
            gold_answer=_answer_from_anchor(ch_var_def["text"], "Value-at-Risk is a measure", max_len=320),
            gold_citations=[_citation_from_chunk(ch_var_def, anchor="Value-at-Risk")],
            topic="var",
            difficulty="easy",
        ),
        DS1Example(
            id="ds1_pub_003",
            question="What are the three basic components of RiskMetrics?",
            gold_answer=_answer_from_anchor(ch_rm_components["text"], "RiskMetrics has three basic components", max_len=360),
            gold_citations=[_citation_from_chunk(ch_rm_components, anchor="three basic components")],
            topic="riskmetrics_overview",
            difficulty="easy",
        ),
        DS1Example(
            id="ds1_pub_004",
            question="In the delta approximation, what does Delta represent?",
            gold_answer=_answer_from_anchor(ch_delta_def["text"], "Delta is the first derivative", max_len=320),
            gold_citations=[_citation_from_chunk(ch_delta_def, anchor="Delta is the first derivative")],
            topic="options_greeks",
            difficulty="medium",
        ),
        DS1Example(
            id="ds1_pub_005",
            question="What are the three major steps of the Monte Carlo methodology described by RiskMetrics?",
            gold_answer=_answer_from_anchor(ch_mc_steps["text"], "The Monte Carlo methodology consists of three major steps", max_len=360),
            gold_citations=[_citation_from_chunk(ch_mc_steps, anchor="three major steps")],
            topic="monte_carlo",
            difficulty="medium",
        ),
        DS1Example(
            id="ds1_pub_006",
            question="What confidence level does the BCBS market risk standard require for Expected Shortfall (ES)?",
            gold_answer=_answer_from_anchor(ch_es_min_standards["text"], "97.5th percentile", max_len=320),
            gold_citations=[_citation_from_chunk(ch_es_min_standards, anchor="97.5th percentile")],
            topic="frtb_es",
            difficulty="easy",
        ),
        DS1Example(
            id="ds1_pub_007",
            question="What is a Geometric Brownian Motion (GBM) as defined in the Black-Scholes model paper?",
            gold_answer=_answer_from_anchor(ch_gbm_def["text"], "A Geometric Brownian Motion", max_len=360),
            gold_citations=[_citation_from_chunk(ch_gbm_def, anchor="Geometric Brownian Motion")],
            topic="stochastic_processes",
            difficulty="medium",
        ),
        DS1Example(
            id="ds1_pub_008",
            question="What is Ito's lemma, and what does it provide?",
            gold_answer=_answer_from_anchor(ch_ito_def["text"], "Ito’s lemma is an identity", max_len=360),
            gold_citations=[_citation_from_chunk(ch_ito_def, anchor="Ito’s lemma is an identity")],
            topic="stochastic_calculus",
            difficulty="medium",
        ),
        DS1Example(
            id="ds1_pub_009",
            question="List key assumptions stated for the underlying asset and market environment in the Black-Scholes paper.",
            gold_answer=_answer_from_anchor(ch_bs_assumptions["text"], "characteristics", max_len=360),
            gold_citations=[_citation_from_chunk(ch_bs_assumptions, anchor="characteristics")],
            topic="black_scholes_assumptions",
            difficulty="medium",
        ),
        DS1Example(
            id="ds1_pub_010",
            question="Which numerical method is used to approximate option pricing PDE solutions in the stochastic volatility paper (Hao et al.)?",
            gold_answer=_answer_from_anchor(ch_fd_methods["text"], "finite difference methods", max_len=320),
            gold_citations=[_citation_from_chunk(ch_fd_methods, anchor="finite difference methods")],
            topic="numerical_methods",
            difficulty="easy",
        ),
    ]

    ds1.examples = ds1_examples
    ds1.__init__(**ds1.model_dump())

    # ---------------------------------------------------------------------
    # DS2 (Retrieval qrels)
    # ---------------------------------------------------------------------
    ds2 = gen.create_empty_ds2(description="Production DS2 qrels for public corpus (real chunk_id)")

    def qrel(ch: Dict[str, Any], rel: int) -> DS2Qrel:
        ps = ch["page_span"]
        return DS2Qrel(
            chunk_id=ch["id"],
            relevance=rel,
            doc_id=ch["doc_id"],
            page_span=PageSpan(
                start_page=int(ps["start_page"]),
                end_page=int(ps["end_page"]),
                start_char=ps.get("start_char"),
                end_char=ps.get("end_char"),
            ),
        )

    ds2_examples: List[DS2Example] = [
        DS2Example(
            id="ds2_pub_001",
            query="Definition of Value-at-Risk (VaR)",
            qrels=[qrel(ch_var_def, 2)],
            query_type="definition",
        ),
        DS2Example(
            id="ds2_pub_002",
            query="RiskMetrics definition of risk",
            qrels=[qrel(ch_risk_def, 2)],
            query_type="definition",
        ),
        DS2Example(
            id="ds2_pub_003",
            query="RiskMetrics three basic components",
            qrels=[qrel(ch_rm_components, 2)],
            query_type="definition",
        ),
        DS2Example(
            id="ds2_pub_004",
            query="Delta approximation definition (first derivative of option price)",
            qrels=[qrel(ch_delta_def, 2)],
            query_type="definition",
        ),
        DS2Example(
            id="ds2_pub_005",
            query="Monte Carlo methodology steps RiskMetrics",
            qrels=[qrel(ch_mc_steps, 2)],
            query_type="procedure",
        ),
        DS2Example(
            id="ds2_pub_006",
            query="Expected Shortfall confidence level 97.5 one-tailed",
            qrels=[qrel(ch_es_min_standards, 2)],
            query_type="definition",
        ),
        DS2Example(
            id="ds2_pub_007",
            query="Geometric Brownian Motion definition and SDE",
            qrels=[qrel(ch_gbm_def, 2)],
            query_type="definition",
        ),
        DS2Example(
            id="ds2_pub_008",
            query="Ito's lemma definition and formula",
            qrels=[qrel(ch_ito_def, 2)],
            query_type="definition",
        ),
        DS2Example(
            id="ds2_pub_009",
            query="Black-Scholes assumptions risk-free rate no dividends no transaction costs",
            qrels=[qrel(ch_bs_assumptions, 2)],
            query_type="procedure",
        ),
        DS2Example(
            id="ds2_pub_010",
            query="finite difference methods for option pricing PDE",
            qrels=[qrel(ch_fd_methods, 2)],
            query_type="methods",
        ),
    ]

    ds2.examples = ds2_examples
    ds2.__init__(**ds2.model_dump())

    # ---------------------------------------------------------------------
    # DS3 (Unanswerable / traps)
    # ---------------------------------------------------------------------
    ds3 = gen.create_empty_ds3(description="Production DS3 unanswerable/traps for public corpus")

    ds3.examples = [
        DS3Example(
            id="ds3_pub_001",
            question="What was the closing price of the S&P 500 index on 2025-12-20?",
            reason_unanswerable="Public corpus contains methodology and papers, not live market data or historical index levels for specific dates.",
            trap_type="out_of_scope",
            expected_behavior="refuse",
        ),
        DS3Example(
            id="ds3_pub_002",
            question="Give me today's implied volatility surface for EURUSD and cite sources.",
            reason_unanswerable="Corpus does not include real-time market data feeds or current implied vol surfaces.",
            trap_type="out_of_scope",
            expected_behavior="refuse",
        ),
        DS3Example(
            id="ds3_pub_003",
            question="Which trading strategy will guarantee profit with options?",
            reason_unanswerable="Asks for investment advice and guarantees; not answerable from the technical corpus and unsafe to provide.",
            trap_type="ambiguous",
            expected_behavior="refuse",
        ),
        DS3Example(
            id="ds3_pub_004",
            question="Is VaR always better than Expected Shortfall? Answer yes or no.",
            reason_unanswerable="Question is overly reductive; depends on context, assumptions, and regulatory objectives. Requires clarification.",
            trap_type="ambiguous",
            expected_behavior="clarify",
        ),
        DS3Example(
            id="ds3_pub_005",
            question="What does the BCBS FRTB standard say about crypto market risk in 2024?",
            reason_unanswerable="The corpus includes a 2019 BCBS market risk standard; it does not necessarily cover crypto-specific 2024 guidance.",
            trap_type="temporal_mismatch",
            expected_behavior="flag_uncertainty",
        ),
        DS3Example(
            id="ds3_pub_006",
            question="In the paper 'Some Contributions to Sequential Monte Carlo Methods for Option Pricing', what is the BTC funding rate in 2023?",
            reason_unanswerable="Mixes an academic topic with unrelated market data not present in the corpus.",
            trap_type="conflicting_assumptions",
            expected_behavior="refuse",
        ),
        DS3Example(
            id="ds3_pub_007",
            question="How do I compute the Vega of a plain vanilla interest rate swap?",
            reason_unanswerable="Plain vanilla swaps are not options; 'Vega' is an options volatility sensitivity. The term usage is confused.",
            trap_type="similar_term_confusion",
            expected_behavior="clarify",
            trap_answer="Differentiate the swap PV with respect to implied volatility.",
        ),
        DS3Example(
            id="ds3_pub_008",
            question="What is the Black-Scholes formula for an American call with early exercise and dividends, exactly as in the standard model?",
            reason_unanswerable="American options with dividends generally do not have the same closed-form as the standard European Black-Scholes; the premise is inconsistent.",
            trap_type="conflicting_assumptions",
            expected_behavior="flag_uncertainty",
        ),
        DS3Example(
            id="ds3_pub_009",
            question="Which exact page in 'RiskMetrics Technical Document' contains a complete list of all daily volatilities for 2025?",
            reason_unanswerable="The corpus contains methodological descriptions, not a 2025 daily dataset listing embedded in the PDF pages.",
            trap_type="out_of_scope",
            expected_behavior="refuse",
        ),
        DS3Example(
            id="ds3_pub_010",
            question="What is the best delta for hedging any option position?",
            reason_unanswerable="Delta hedging depends on the option, underlying dynamics, hedging frequency, costs, and risk preferences; needs clarification.",
            trap_type="ambiguous",
            expected_behavior="clarify",
        ),
    ]
    ds3.__init__(**ds3.model_dump())

    # ---------------------------------------------------------------------
    # DS4 (Multi-hop)
    # ---------------------------------------------------------------------
    ds4 = gen.create_empty_ds4(description="Production DS4 multi-hop for public corpus (real citations)")

    ds4.examples = [
        DS4Example(
            id="ds4_pub_001",
            question="Compare Value-at-Risk (VaR) and Expected Shortfall (ES): what does VaR represent in RiskMetrics, and what confidence level is specified for ES in the BCBS market risk standard?",
            gold_answer=(
                "RiskMetrics describes VaR as a measure of the maximum potential change in value of a portfolio "
                "with a given probability over a preset horizon. The BCBS market risk standard specifies that ES "
                "must be computed using a 97.5th percentile, one-tailed confidence level."
            ),
            gold_citations=[
                _citation_from_chunk(ch_var_def, anchor="Value-at-Risk"),
                _citation_from_chunk(ch_es_min_standards, anchor="97.5th percentile"),
            ],
            hops=[
                DS4HopInfo(
                    hop_number=1,
                    sub_question="How is Value-at-Risk (VaR) defined in RiskMetrics?",
                    required_doc_ids=["riskmetrics_1996"],
                    required_concepts=["VaR", "probability", "horizon"],
                ),
                DS4HopInfo(
                    hop_number=2,
                    sub_question="What confidence level is required for Expected Shortfall (ES) in BCBS FRTB?",
                    required_doc_ids=["bcbs_frtb_2019"],
                    required_concepts=["ES", "97.5th percentile", "one-tailed"],
                ),
            ],
            reasoning_type="comparative",
            min_required_sources=2,
            difficulty="hard",
        ),
        DS4Example(
            id="ds4_pub_002",
            question="Summarize two numerical approaches mentioned across the corpus: what are RiskMetrics' three major steps of Monte Carlo simulation, and which numerical method is used to approximate option pricing PDE solutions in Hao et al. (2024)?",
            gold_answer=(
                "RiskMetrics describes Monte Carlo simulation as involving scenario generation, portfolio valuation "
                "under those scenarios, and summarizing the results (e.g., as a distribution or risk measure). "
                "Hao et al. (2024) describe approximating option pricing PDE solutions numerically using finite "
                "difference methods."
            ),
            gold_citations=[
                _citation_from_chunk(ch_mc_steps, anchor="three major steps"),
                _citation_from_chunk(ch_fd_methods, anchor="finite difference methods"),
            ],
            hops=[
                DS4HopInfo(
                    hop_number=1,
                    sub_question="What are the major steps of Monte Carlo simulation in RiskMetrics?",
                    required_doc_ids=["riskmetrics_1996"],
                    required_concepts=["Monte Carlo", "scenario generation", "revaluation"],
                ),
                DS4HopInfo(
                    hop_number=2,
                    sub_question="What numerical method is used for PDE approximation in Hao et al. (2024)?",
                    required_doc_ids=["arxiv_2408_15416v1"],
                    required_concepts=["PDE", "finite difference"],
                ),
            ],
            reasoning_type="comparative",
            min_required_sources=2,
            difficulty="hard",
        ),
    ]
    ds4.__init__(**ds4.model_dump())

    # ---------------------------------------------------------------------
    # DS5 (Structured extraction)
    # ---------------------------------------------------------------------
    ds5 = gen.create_empty_ds5(description="Production DS5 structured extraction for public corpus (real schema/output)")

    # DS5 helper: parse simple bullet lists (PDF text uses "•" or leading dashes/newlines)
    def extract_bullets(text: str, *, after_anchor: str, max_items: int = 10) -> List[str]:
        t = text
        idx = t.lower().find(after_anchor.lower())
        if idx >= 0:
            t = t[idx + len(after_anchor) :]
        # Split into lines; keep lines that look like bullets.
        items: List[str] = []
        for line in t.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("•"):
                items.append(_norm_ws(s.lstrip("•").strip()))
            if len(items) >= max_items:
                break
        return items

    rm_components = extract_bullets(ch_rm_components["text"], after_anchor="three basic components")
    bs_assumptions = extract_bullets(ch_bs_assumptions["text"], after_anchor="characteristics")

    ds5.examples = [
        DS5Example(
            id="ds5_pub_001",
            question="Extract the three basic components of RiskMetrics as a JSON list.",
            structured_output=DS5StructuredOutput(
                extraction_type="classification",
                output_schema={
                    "type": "object",
                    "properties": {
                        "components": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["components"],
                },
                gold_output={"components": rm_components},
            ),
            gold_citations=[_citation_from_chunk(ch_rm_components, anchor="three basic components")],
            required_fields=["components"],
        ),
        DS5Example(
            id="ds5_pub_002",
            question="Extract the Black-Scholes market/underlying assumptions listed in the paper as JSON.",
            structured_output=DS5StructuredOutput(
                extraction_type="assumptions",
                output_schema={
                    "type": "object",
                    "properties": {
                        "assumptions": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["assumptions"],
                },
                gold_output={"assumptions": bs_assumptions},
            ),
            gold_citations=[_citation_from_chunk(ch_bs_assumptions, anchor="characteristics")],
            required_fields=["assumptions"],
        ),
        DS5Example(
            id="ds5_pub_003",
            question="Extract the confidence level requirement for Expected Shortfall (ES) from the BCBS standard.",
            structured_output=DS5StructuredOutput(
                extraction_type="parameter_ranges",
                output_schema={
                    "type": "object",
                    "properties": {
                        "confidence_level": {"type": "number"},
                        "tail": {"type": "string"},
                    },
                    "required": ["confidence_level", "tail"],
                },
                gold_output={"confidence_level": 97.5, "tail": "one-tailed"},
            ),
            gold_citations=[_citation_from_chunk(ch_es_min_standards, anchor="97.5th percentile")],
            required_fields=["confidence_level", "tail"],
        ),
        DS5Example(
            id="ds5_pub_004",
            question="Extract the Geometric Brownian Motion (GBM) SDE parameters mentioned in the definition as JSON.",
            structured_output=DS5StructuredOutput(
                extraction_type="formula",
                output_schema={
                    "type": "object",
                    "properties": {
                        "sde": {"type": "string"},
                        "mu_symbol": {"type": "string"},
                        "sigma_symbol": {"type": "string"},
                        "wiener_symbol": {"type": "string"},
                    },
                    "required": ["sde", "mu_symbol", "sigma_symbol", "wiener_symbol"],
                },
                gold_output={
                    "sde": "dSt = μSt dt + σSt dWt",
                    "mu_symbol": "μ",
                    "sigma_symbol": "σ",
                    "wiener_symbol": "Wt",
                },
            ),
            gold_citations=[_citation_from_chunk(ch_gbm_def, anchor="dSt")],
            required_fields=["sde", "mu_symbol", "sigma_symbol", "wiener_symbol"],
        ),
    ]
    ds5.__init__(**ds5.model_dump())

    # ---------------------------------------------------------------------
    # Save + validate
    # ---------------------------------------------------------------------
    save_dataset(ds1, out_dir / "ds1_factual_qa_public.json")
    save_dataset(ds2, out_dir / "ds2_retrieval_qrels_public.json")
    save_dataset(ds3, out_dir / "ds3_unanswerable_traps_public.json")
    save_dataset(ds4, out_dir / "ds4_multihop_public.json")
    save_dataset(ds5, out_dir / "ds5_structured_extraction_public.json")

    print("✓ Generated production datasets:")
    for p in [
        "ds1_factual_qa_public.json",
        "ds2_retrieval_qrels_public.json",
        "ds3_unanswerable_traps_public.json",
        "ds4_multihop_public.json",
        "ds5_structured_extraction_public.json",
    ]:
        print(f"  - {out_dir / p}")

    print("\nValidation summaries:")
    print("  DS1:", validator.validate_ds1(ds1).get_summary().replace("\n", " | "))
    print("  DS2:", validator.validate_ds2(ds2).get_summary().replace("\n", " | "))
    print("  DS3:", validator.validate_ds3(ds3).get_summary().replace("\n", " | "))
    print("  DS4:", validator.validate_ds4(ds4).get_summary().replace("\n", " | "))
    print("  DS5:", validator.validate_ds5(ds5).get_summary().replace("\n", " | "))


def main() -> int:
    index_strategy = "fixed"
    if len(sys.argv) > 1:
        index_strategy = sys.argv[1]
    build_production_datasets(index_strategy=index_strategy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


