# =============================================================
# summarizer.py  —  Step 2: Extractive Summarization
# Models: TextRank (graph-based), TF-IDF weighted extraction,
#         LSA (Latent Semantic Analysis) via TruncatedSVD
# =============================================================

import re
import math
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from preprocessor import (
    sent_tokenize, word_tokenize, remove_stopwords,
    word_freq_scores, position_scores, clean_text
)

# =============================================================
# 1. TEXTRANK SUMMARIZER (Graph-based, similar to PageRank)
# =============================================================

def textrank_summarize(text, num_sentences=5, damping=0.85, iterations=30):
    """
    TextRank algorithm:
    - Build similarity graph of sentences
    - Run PageRank-style iteration
    - Return top-scored sentences
    """
    sentences = sent_tokenize(clean_text(text))
    if len(sentences) < 3:
        return text[:1000], sentences

    # Clamp
    num_sentences = min(num_sentences, max(1, len(sentences) // 3))

    # Build TF-IDF matrix for sentence similarity
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sim_matrix = cosine_similarity(tfidf_matrix)
    except Exception:
        # Fallback: word overlap similarity
        sim_matrix = _word_overlap_matrix(sentences)

    # Normalize similarity matrix
    np.fill_diagonal(sim_matrix, 0)
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    sim_matrix = sim_matrix / row_sums

    # PageRank iteration
    n = len(sentences)
    scores = np.ones(n) / n
    for _ in range(iterations):
        new_scores = (1 - damping) / n + damping * sim_matrix.T.dot(scores)
        if np.allclose(scores, new_scores, atol=1e-4):
            break
        scores = new_scores

    # Blend with position scores
    pos_sc = position_scores(sentences)
    for i, s in enumerate(sentences):
        scores[i] = 0.7 * scores[i] + 0.3 * pos_sc.get(s, 0.5)

    # Pick top sentences, preserve original order
    top_indices = sorted(
        np.argsort(scores)[-num_sentences:].tolist()
    )
    summary_sentences = [sentences[i] for i in top_indices]
    summary = ' '.join(summary_sentences)
    return summary, sentences

def _word_overlap_matrix(sentences):
    """Fallback: Jaccard similarity between sentence word sets."""
    n = len(sentences)
    matrix = np.zeros((n, n))
    token_sets = [set(remove_stopwords(word_tokenize(s))) for s in sentences]
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(token_sets[i] & token_sets[j])
            union = len(token_sets[i] | token_sets[j])
            score = inter / union if union > 0 else 0
            matrix[i][j] = matrix[j][i] = score
    return matrix

# =============================================================
# 2. LSA SUMMARIZER (Latent Semantic Analysis)
# Uses TF-IDF + TruncatedSVD to find latent topics
# =============================================================

def lsa_summarize(text, num_sentences=5, n_components=5):
    """
    LSA-based extractive summarization:
    - TF-IDF vectorize sentences
    - Apply SVD to find latent semantic axes
    - Score sentences by their importance across all topics
    """
    sentences = sent_tokenize(clean_text(text))
    if len(sentences) < 4:
        return text[:800], sentences

    num_sentences = min(num_sentences, len(sentences) // 3)
    n_components = min(n_components, len(sentences) - 1, 20)

    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
        tfidf_matrix = vectorizer.fit_transform(sentences)

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_matrix = svd.fit_transform(tfidf_matrix)  # (n_sentences, n_components)

        # Score = L2 norm across all latent dimensions
        scores = np.linalg.norm(svd_matrix, axis=1)

        # Blend with position bias
        pos_sc = position_scores(sentences)
        for i, s in enumerate(sentences):
            scores[i] = 0.65 * scores[i] + 0.35 * pos_sc.get(s, 0.5)

        top_indices = sorted(np.argsort(scores)[-num_sentences:].tolist())
        summary = ' '.join(sentences[i] for i in top_indices)
        return summary, sentences

    except Exception as e:
        # Fallback to TextRank
        return textrank_summarize(text, num_sentences)

# =============================================================
# 3. FREQUENCY-BASED SUMMARIZER (TF-IDF word scores)
# =============================================================

def frequency_summarize(text, num_sentences=5):
    """
    TF-IDF word frequency summarizer:
    - Score each word by TF-IDF importance
    - Score sentences as average of word scores
    - Return top sentences
    """
    sentences = sent_tokenize(clean_text(text))
    if not sentences:
        return text[:500], []

    num_sentences = min(num_sentences, max(1, len(sentences) // 3))
    word_scores = word_freq_scores(text)

    sentence_scores = {}
    for sent in sentences:
        tokens = remove_stopwords(word_tokenize(sent))
        if not tokens:
            continue
        score = sum(word_scores.get(t, 0) for t in tokens) / len(tokens)
        sentence_scores[sent] = score

    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    # Restore original order
    ordered = [s for s in sentences if s in set(top_sentences)]
    return ' '.join(ordered), sentences

# =============================================================
# 4. SECTION-AWARE SUMMARY
# =============================================================

def section_aware_summarize(sections_dict, sentences_per_section=2):
    """
    Summarize each section independently and combine.
    Useful for structured papers.
    """
    priority_order = [
        "abstract", "introduction", "methodology", "methods",
        "results", "conclusion", "conclusions", "discussion"
    ]

    combined = []
    used_sections = []

    # Process priority sections first
    for sec in priority_order:
        for key, content in sections_dict.items():
            if sec in key.lower() and content and len(content) > 100:
                summary, _ = frequency_summarize(content, num_sentences=sentences_per_section)
                if summary:
                    combined.append(f"[{key.title()}] {summary}")
                    used_sections.append(key)
                break

    return '\n\n'.join(combined)

# =============================================================
# 5. KEYWORD EXTRACTION (TF-IDF based)
# =============================================================

def extract_keywords(text, top_n=15):
    """Extract important keywords using TF-IDF scoring."""
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        sentences = [text]

    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=500,
            ngram_range=(1, 3),
            min_df=1
        )
        matrix = vectorizer.fit_transform(sentences)
        scores = np.asarray(matrix.mean(axis=0)).flatten()
        feature_names = vectorizer.get_feature_names_out()

        top_indices = np.argsort(scores)[-top_n * 2:][::-1]
        # Filter: prefer multi-word, remove pure numbers
        keywords = []
        for i in top_indices:
            term = feature_names[i]
            if not term.isdigit() and len(term) > 2:
                keywords.append((term, float(scores[i])))
            if len(keywords) >= top_n:
                break
        return keywords
    except Exception:
        # Fallback: word frequency
        ws = word_freq_scores(text)
        return sorted(ws.items(), key=lambda x: x[1], reverse=True)[:top_n]
