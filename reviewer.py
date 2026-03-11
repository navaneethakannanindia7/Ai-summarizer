# =============================================================
# reviewer.py  —  Step 4: Automated Paper Quality Scoring
# Models: TF-IDF features + heuristic ML scoring
# Dimensions: Novelty, Clarity, Methodology, Evidence, Impact
# =============================================================

import re
import math
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessor import (
    sent_tokenize, word_tokenize, remove_stopwords,
    clean_text, detect_sections, SECTION_HEADERS
)

# ── Scoring vocabulary lists ──────────────────────────────────

NOVELTY_TERMS = [
    "novel", "new", "first", "introduce", "propose", "innovative",
    "unprecedented", "pioneer", "breakthrough", "original", "unique",
    "state-of-the-art", "state of the art", "outperform", "surpass",
    "advance", "improve upon", "better than", "superior"
]

RIGOR_TERMS = [
    "statistical", "significance", "p-value", "confidence interval",
    "baseline", "ablation", "cross-validation", "benchmark",
    "reproducible", "controlled", "hypothesis", "empirical",
    "quantitative", "qualitative", "analysis", "experiment",
    "evaluation", "comparison", "dataset", "ground truth"
]

CLARITY_NEGATIVE = [
    "unclear", "ambiguous", "vague", "confusing", "inconsistent",
    "typo", "grammar", "notation", "undefined"
]

IMPACT_TERMS = [
    "application", "real-world", "practical", "deploy", "scalable",
    "industry", "production", "useful", "benefit", "significant",
    "important", "relevant", "broader impact", "societal"
]

SECTION_WEIGHTS = {
    "abstract": 1.3,
    "introduction": 1.2,
    "methodology": 1.4,
    "methods": 1.4,
    "results": 1.3,
    "conclusion": 1.2,
    "related work": 0.9,
    "experiments": 1.3,
    "evaluation": 1.2
}


# =============================================================
# MAIN SCORER
# =============================================================

def score_paper(text, metadata=None):
    """
    Compute multi-dimensional quality scores for a research paper.
    Returns dict with scores (0-10) and explanations.
    """
    sections = detect_sections(text)
    sentences = sent_tokenize(clean_text(text))
    full_lower = text.lower()
    word_count = len(word_tokenize(text))

    scores = {
        "novelty":       _score_novelty(text, sentences),
        "clarity":       _score_clarity(text, sentences, sections),
        "methodology":   _score_methodology(text, sections),
        "evidence":      _score_evidence(text, sections),
        "impact":        _score_impact(text, sentences),
        "completeness":  _score_completeness(sections, word_count),
        "writing":       _score_writing_quality(sentences),
    }

    # Overall weighted average
    weights = {
        "novelty": 0.20,
        "clarity": 0.15,
        "methodology": 0.25,
        "evidence": 0.20,
        "impact": 0.10,
        "completeness": 0.05,
        "writing": 0.05
    }
    overall = sum(scores[k]["score"] * weights[k] for k in scores)
    scores["overall"] = round(overall, 2)

    # Recommendation
    scores["recommendation"] = _get_recommendation(overall, scores)
    scores["strengths"] = _extract_strengths(scores, text)
    scores["weaknesses"] = _extract_weaknesses(scores, text, sections)
    scores["paper_type"] = _classify_paper_type(text, sections)
    scores["reading_difficulty"] = _estimate_difficulty(text)

    return scores


# =============================================================
# DIMENSION SCORERS
# =============================================================

def _score_novelty(text, sentences):
    lower = text.lower()
    hits = sum(lower.count(t) for t in NOVELTY_TERMS)
    total_words = max(len(word_tokenize(text)), 1)

    # Density of novelty terms
    density = min(hits / (total_words / 100), 10)
    score = min(10, 2 + density * 2.5)

    # Boost if claims explicit comparisons
    if re.search(r'(outperform|improve|better than|exceed|surpass)', lower):
        score = min(10, score + 1.5)

    justification = (
        f"Found {hits} novelty-indicating terms. "
        + ("Paper makes explicit performance comparisons. " if re.search(r'outperform|surpass', lower) else "")
        + ("Consider adding stronger novelty statements." if score < 6 else "Strong novelty signaling.")
    )
    return {"score": round(score, 1), "justification": justification}


def _score_clarity(text, sentences, sections):
    score = 7.0  # Start at 7 and adjust

    # Penalize very short or very long sentences
    lengths = [len(s.split()) for s in sentences]
    avg_len = np.mean(lengths) if lengths else 20
    if avg_len > 45:
        score -= 1.5  # Too verbose
    elif avg_len < 10:
        score -= 1.0  # Too terse
    elif 15 <= avg_len <= 30:
        score += 0.5  # Ideal range

    # Has defined structure?
    section_coverage = sum(1 for k in sections if any(sh in k for sh in SECTION_HEADERS))
    score += min(1.5, section_coverage * 0.3)

    # Penalize very long papers (harder to read)
    word_count = len(word_tokenize(text))
    if word_count > 15000:
        score -= 0.5

    score = max(1, min(10, score))
    justification = (
        f"Average sentence length: {avg_len:.0f} words. "
        f"Detected {section_coverage} standard sections. "
        + ("Well-structured document." if section_coverage >= 4 else "Consider adding clearer section headers.")
    )
    return {"score": round(score, 1), "justification": justification}


def _score_methodology(text, sections):
    lower = text.lower()
    score = 4.0

    # Check for methodology section
    has_method_section = any('method' in k or 'approach' in k for k in sections)
    if has_method_section:
        score += 2.0

    # Rigor indicators
    rigor_hits = sum(lower.count(t) for t in RIGOR_TERMS)
    score += min(3, rigor_hits * 0.15)

    # Quantitative results
    has_numbers = len(re.findall(r'\d+\.?\d*\s*%', text)) > 3
    if has_numbers:
        score += 0.8

    # Ablation study
    if 'ablation' in lower:
        score += 0.5

    # Reproducibility markers
    if any(t in lower for t in ['code available', 'github', 'open source', 'released']):
        score += 0.5

    score = max(1, min(10, score))
    justification = (
        f"Rigour indicators found: {rigor_hits}. "
        + ("Methodology section present. " if has_method_section else "No dedicated methodology section detected. ")
        + ("Quantitative results present. " if has_numbers else "")
        + ("Ablation study mentioned. " if 'ablation' in lower else "")
    )
    return {"score": round(score, 1), "justification": justification}


def _score_evidence(text, sections):
    lower = text.lower()
    score = 4.0

    # Count tables and figures
    fig_count = len(re.findall(r'\b(?:figure|fig\.?)\s*\d+', lower))
    tab_count = len(re.findall(r'\btable\s*\d+', lower))
    eq_count = len(re.findall(r'\bequation\s*\d+|\beq\.\s*\d+', lower))

    score += min(2.5, fig_count * 0.3)
    score += min(1.5, tab_count * 0.4)
    score += min(0.5, eq_count * 0.1)

    # References count
    ref_matches = re.findall(r'\[\d+\]', text)
    ref_count = len(set(ref_matches))
    if ref_count > 20:
        score += 1.0
    elif ref_count > 10:
        score += 0.5

    # Statistical tests
    if any(t in lower for t in ['p-value', 'p < 0', 'p=0', 'statistical', 'significance']):
        score += 0.5

    score = max(1, min(10, score))
    justification = (
        f"Figures: ~{fig_count}, Tables: ~{tab_count}, Equations: ~{eq_count}. "
        f"~{ref_count} reference citations found. "
        + ("Statistical significance testing used." if 'p-value' in lower or 'p < ' in lower else "")
    )
    return {"score": round(score, 1), "justification": justification}


def _score_impact(text, sentences):
    lower = text.lower()
    hits = sum(lower.count(t) for t in IMPACT_TERMS)
    score = min(10, 3 + hits * 0.4)

    # Has real-world application examples?
    if re.search(r'real.world|production|deploy|industry|patient|clinical', lower):
        score = min(10, score + 1.5)

    justification = (
        f"Impact/application terms found: {hits}. "
        + ("Real-world applications discussed." if re.search(r'real.world|deploy', lower) else "Consider adding practical application discussion.")
    )
    return {"score": round(score, 1), "justification": justification}


def _score_completeness(sections, word_count):
    essential = ["abstract", "introduction", "conclusion"]
    important = ["method", "results", "related", "experiment"]

    essential_found = sum(1 for e in essential if any(e in k for k in sections))
    important_found = sum(1 for i in important if any(i in k for k in sections))

    score = (essential_found / len(essential)) * 5 + (important_found / len(important)) * 3

    # Word count bonus
    if 4000 < word_count < 12000:
        score += 1.5
    elif word_count >= 12000:
        score += 0.5

    score = max(1, min(10, score))
    return {
        "score": round(score, 1),
        "justification": f"Essential sections: {essential_found}/3, Important sections: {important_found}/4. Word count: ~{word_count}."
    }


def _score_writing_quality(sentences):
    if not sentences:
        return {"score": 5.0, "justification": "Unable to assess."}

    score = 7.0
    lengths = [len(s.split()) for s in sentences]

    # Penalize variance extremes
    std = np.std(lengths)
    if std > 25:
        score -= 1.0  # Too variable
    elif std < 5:
        score -= 0.5  # Too uniform (robotic)

    # Passive voice density
    passive = sum(1 for s in sentences if re.search(r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', s))
    passive_ratio = passive / len(sentences)
    if passive_ratio > 0.4:
        score -= 0.5

    score = max(1, min(10, score))
    return {
        "score": round(score, 1),
        "justification": f"Avg sentence length: {np.mean(lengths):.0f} words (std: {std:.1f}). Passive voice: {passive_ratio:.0%} of sentences."
    }


# =============================================================
# HELPERS
# =============================================================

def _get_recommendation(overall, scores):
    if overall >= 8.0:
        return {"decision": "Accept", "color": "green", "confidence": "High"}
    elif overall >= 6.5:
        return {"decision": "Accept with Minor Revisions", "color": "blue", "confidence": "Medium"}
    elif overall >= 5.0:
        return {"decision": "Major Revisions Required", "color": "orange", "confidence": "Medium"}
    else:
        return {"decision": "Reject / Major Rework", "color": "red", "confidence": "Low"}


def _extract_strengths(scores, text):
    strengths = []
    lower = text.lower()
    if scores["novelty"]["score"] >= 7:
        strengths.append("Strong novelty claims with clear differentiation from prior work")
    if scores["methodology"]["score"] >= 7:
        strengths.append("Solid methodological rigor with quantitative evaluation")
    if scores["evidence"]["score"] >= 7:
        strengths.append("Well-supported claims with figures, tables, and citations")
    if 'ablation' in lower:
        strengths.append("Includes ablation study to validate design choices")
    if re.search(r'github|code available|open source', lower):
        strengths.append("Code/data availability improves reproducibility")
    if scores["clarity"]["score"] >= 7:
        strengths.append("Clear and well-structured presentation")
    return strengths if strengths else ["Paper presents research in a structured manner"]


def _extract_weaknesses(scores, text, sections):
    weaknesses = []
    if scores["novelty"]["score"] < 6:
        weaknesses.append("Novelty could be stated more explicitly; differentiation from existing work unclear")
    if scores["methodology"]["score"] < 6:
        weaknesses.append("Methodology section needs more detail; reproducibility may be limited")
    if scores["evidence"]["score"] < 5:
        weaknesses.append("Limited empirical evidence; more experiments or baselines would strengthen claims")
    if scores["completeness"]["score"] < 6:
        weaknesses.append("Paper structure is incomplete; missing essential sections")
    if not any('related' in k or 'literature' in k for k in sections):
        weaknesses.append("No dedicated related work section; contextualizing contributions is important")
    if scores["impact"]["score"] < 5:
        weaknesses.append("Practical impact/applications not adequately discussed")
    return weaknesses if weaknesses else ["No critical weaknesses identified at this analysis level"]


def _classify_paper_type(text, sections):
    lower = text.lower()
    if 'survey' in lower[:2000] or 'review' in lower[:2000] or 'overview' in lower[:2000]:
        return "Survey / Review"
    if any('dataset' in lower[:3000], 'benchmark' in lower[:3000], 'annotation' in lower[:3000]):
        return "Dataset / Benchmark"
    if 'theorem' in lower or 'proof' in lower or 'lemma' in lower:
        return "Theoretical"
    if re.search(r'demo|demonstration|system|tool|interface', lower[:3000]):
        return "System / Demo"
    return "Empirical Research"


def _estimate_difficulty(text):
    """Estimate technical difficulty based on vocabulary richness."""
    tokens = word_tokenize(text[:5000])
    unique = len(set(tokens))
    total = len(tokens) + 1
    ttr = unique / total  # Type-Token Ratio

    # Check for technical symbols
    math_density = len(re.findall(r'[α-ωΑ-Ω∑∏∫∂∇≈≤≥∈∀∃]|\\[a-z]+\{', text)) / max(len(text), 1)

    if ttr > 0.6 or math_density > 0.002:
        return "Expert"
    elif ttr > 0.45:
        return "Advanced"
    elif ttr > 0.30:
        return "Intermediate"
    else:
        return "Accessible"
