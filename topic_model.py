# =============================================================
# topic_model.py  —  Step 3: Topic Modeling
# Models: LDA (Latent Dirichlet Allocation), NMF
# =============================================================

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.preprocessing import normalize
from preprocessor import sent_tokenize, word_tokenize, remove_stopwords, clean_text

# ── Research domain taxonomy ─────────────────────────────────
DOMAIN_KEYWORDS = {
    "Machine Learning / AI": [
        "neural", "network", "deep learning", "training", "model", "classification",
        "accuracy", "gradient", "backpropagation", "transformer", "attention",
        "convolutional", "lstm", "reinforcement", "supervised", "unsupervised",
        "generative", "adversarial", "bert", "gpt", "embedding", "feature"
    ],
    "Computer Vision": [
        "image", "pixel", "detection", "segmentation", "recognition", "visual",
        "camera", "video", "object", "feature extraction", "convolution",
        "bounding box", "yolo", "resnet", "vgg", "augmentation", "annotation"
    ],
    "Natural Language Processing": [
        "text", "language", "word", "sentence", "corpus", "semantic",
        "parsing", "sentiment", "translation", "summarization", "token",
        "vocabulary", "syntax", "named entity", "coreference", "dialogue"
    ],
    "Biomedical / Healthcare": [
        "clinical", "patient", "disease", "drug", "genome", "protein",
        "medical", "diagnosis", "treatment", "biomarker", "cancer",
        "brain", "cell", "neural", "mutation", "health", "therapeutic"
    ],
    "Data Mining / Analytics": [
        "database", "query", "mining", "clustering", "anomaly", "pattern",
        "association", "stream", "big data", "hadoop", "spark", "sql",
        "visualization", "statistics", "regression", "correlation"
    ],
    "Robotics / Control": [
        "robot", "control", "sensor", "actuator", "motion", "trajectory",
        "autonomous", "planning", "navigation", "perception", "simulation",
        "kinematics", "feedback", "pid", "manipulation"
    ],
    "Security / Cryptography": [
        "attack", "vulnerability", "encryption", "privacy", "authentication",
        "malware", "intrusion", "firewall", "hash", "blockchain",
        "adversarial", "threat", "certificate", "protocol", "zero-day"
    ],
    "Networks / Systems": [
        "network", "protocol", "bandwidth", "latency", "throughput",
        "distributed", "cloud", "edge", "routing", "tcp", "udp", "packet",
        "server", "client", "fault tolerance", "scalability", "microservice"
    ]
}


# =============================================================
# 1. DOMAIN CLASSIFIER (keyword voting)
# =============================================================

def classify_domain(text, top_n=3):
    """
    Classify paper into research domain using keyword overlap scoring.
    Returns ranked list of (domain, confidence_score).
    """
    text_lower = text.lower()
    tokens = set(word_tokenize(text_lower))
    scores = {}

    for domain, kws in DOMAIN_KEYWORDS.items():
        score = 0
        for kw in kws:
            if ' ' in kw:
                score += 2 * text_lower.count(kw)   # phrase match = higher weight
            else:
                score += tokens.count(kw) if False else text_lower.count(f' {kw} ')
        scores[domain] = score

    total = sum(scores.values()) or 1
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked_pct = [(d, round(100 * s / total, 1)) for d, s in ranked if s > 0]
    return ranked_pct[:top_n]


# =============================================================
# 2. LDA TOPIC MODELING
# =============================================================

def lda_topics(text, n_topics=5, top_words=8):
    """
    Latent Dirichlet Allocation:
    - Discovers probabilistic topics from term co-occurrence
    - Returns list of {topic_id, label, words, weight}
    """
    sentences = sent_tokenize(clean_text(text))
    if len(sentences) < 5:
        sentences = re.split(r'\.\s+', text)

    n_topics = min(n_topics, max(2, len(sentences) // 4))

    try:
        # Use count vectorizer for LDA (not TF-IDF)
        vectorizer = CountVectorizer(
            stop_words='english',
            max_features=1000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        doc_term = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()

        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method='online'
        )
        lda.fit(doc_term)

        topics = []
        for topic_idx, topic_vec in enumerate(lda.components_):
            top_idx = topic_vec.argsort()[-top_words:][::-1]
            words = [feature_names[i] for i in top_idx]
            weight = float(topic_vec[top_idx].mean())
            label = _auto_label_topic(words)
            topics.append({
                "topic_id": topic_idx + 1,
                "label": label,
                "words": words,
                "weight": round(weight, 4)
            })

        # Sort by weight descending
        topics.sort(key=lambda x: x["weight"], reverse=True)
        return topics

    except Exception as e:
        return [{"topic_id": 1, "label": "General", "words": [], "weight": 1.0, "error": str(e)}]


# =============================================================
# 3. NMF TOPIC MODELING (often sharper than LDA)
# =============================================================

def nmf_topics(text, n_topics=5, top_words=8):
    """
    Non-negative Matrix Factorization:
    - Often produces more interpretable topics than LDA
    - Works with TF-IDF input
    """
    sentences = sent_tokenize(clean_text(text))
    if len(sentences) < 4:
        return []

    n_topics = min(n_topics, max(2, len(sentences) // 3))

    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 2)
        )
        tfidf = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()

        nmf = NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=200,
            init='nndsvda'
        )
        nmf.fit(tfidf)

        topics = []
        for i, comp in enumerate(nmf.components_):
            top_idx = comp.argsort()[-top_words:][::-1]
            words = [feature_names[j] for j in top_idx]
            topics.append({
                "topic_id": i + 1,
                "label": _auto_label_topic(words),
                "words": words,
                "weight": round(float(comp.max()), 4)
            })

        topics.sort(key=lambda x: x["weight"], reverse=True)
        return topics

    except Exception as e:
        return []


# =============================================================
# 4. AUTO-LABEL TOPIC
# =============================================================

def _auto_label_topic(words):
    """Generate a human-readable label from topic top words."""
    # Try to use top 2 meaningful words
    stoplike = {'using', 'based', 'method', 'approach', 'paper',
                'propose', 'show', 'result', 'also', 'work'}
    meaningful = [w for w in words if w not in stoplike and len(w) > 3]
    if meaningful:
        label_words = meaningful[:2]
        return ' '.join(w.title() for w in label_words)
    return words[0].title() if words else "Topic"


# =============================================================
# 5. RESEARCH CONTRIBUTION EXTRACTOR
# =============================================================

def extract_contributions(text):
    """
    Extract research contributions using pattern matching.
    Looks for: "we propose", "we present", "this paper introduces", etc.
    """
    contribution_patterns = [
        r'(?:we|this paper|this work|this study)\s+(?:propose|present|introduce|develop|design|create|build|demonstrate|show|describe)\s+([^.!?]{20,150})',
        r'(?:our|the)\s+(?:main|key|primary|novel|proposed|new)\s+(?:contribution|approach|method|model|framework|system)\s+(?:is|are)\s+([^.!?]{20,150})',
        r'(?:the|a)\s+(?:novel|new|proposed|key)\s+([^.!?]{10,100})\s+(?:is|are)\s+(?:proposed|presented|introduced)',
    ]

    contributions = []
    text_lower = text[:15000]  # Check first 15k chars

    for pattern in contribution_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for m in matches:
            contrib = m.group(1).strip()
            contrib = re.sub(r'\s+', ' ', contrib)
            if len(contrib) > 20 and contrib not in contributions:
                contributions.append(contrib)

    return contributions[:5]  # Top 5


# =============================================================
# 6. RESEARCH GAP / LIMITATION EXTRACTOR
# =============================================================

def extract_limitations(text):
    """Extract stated limitations and future work."""
    limit_patterns = [
        r'(?:limitation|drawback|weakness|shortcoming|constraint)s?\s+(?:of|is|are|include|:)\s*([^.!?]{20,200})',
        r'(?:we|our approach)\s+(?:do not|does not|cannot|can\'t|fail|lacks?)\s+([^.!?]{15,150})',
        r'(?:future work|future research|in the future)\s+(?:will|should|could|may|might|can)\s+([^.!?]{20,150})',
        r'(?:it is|this is|one)\s+(?:important|worth)\s+(?:to note|noting|mentioning)\s+that\s+([^.!?]{20,150})',
    ]

    limitations = []
    search_text = text[-5000:] + text[:5000]  # Check beginning and end

    for pattern in limit_patterns:
        for m in re.finditer(pattern, search_text, re.IGNORECASE):
            item = m.group(1).strip()
            item = re.sub(r'\s+', ' ', item)
            if len(item) > 20 and item not in limitations:
                limitations.append(item)

    return limitations[:5]
