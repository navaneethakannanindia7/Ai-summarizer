# =============================================================
# preprocessor.py  —  Step 1: Text Cleaning & Preprocessing
# Uses: regex, string ops (no external NLP library needed)
# =============================================================

import re
import string
import math
from collections import Counter

# ---------- Stop words (built-in, no NLTK needed) ------------
STOP_WORDS = set("""
a about above after again against all also am an and any are aren't as at
be because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further get got had hadn't has hasn't have haven't having he he'd he'll
he's her here here's hers herself him himself his how how's i i'd i'll i'm
i've if in into is isn't it it's its itself let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these
they they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's will with won't
would wouldn't you you'd you'll you're you've your yours yourself yourselves
also et al however thus therefore hence moreover furthermore although
whereas while since whether thus thereby thereby
""".split())

SECTION_HEADERS = {
    "abstract", "introduction", "background", "related work", "literature review",
    "methodology", "methods", "method", "approach", "proposed method",
    "experiments", "experiment", "experimental setup", "evaluation",
    "results", "result", "discussion", "conclusion", "conclusions",
    "future work", "references", "acknowledgements", "acknowledgments",
    "appendix", "supplementary", "contributions", "overview"
}

# ---------- Sentence Tokenizer (regex-based) -----------------
def sent_tokenize(text):
    """Split text into sentences using regex heuristics."""
    # Handle common abbreviations
    text = re.sub(r'\b(Fig|fig|Eq|eq|et|al|i\.e|e\.g|vs|Dr|Mr|Mrs|Prof|etc|approx|ref|refs)\.',
                  lambda m: m.group().replace('.', '<DOT>'), text)
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Restore abbreviation dots
    sentences = [s.replace('<DOT>', '.').strip() for s in sentences]
    return [s for s in sentences if len(s.split()) >= 4]

# ---------- Word Tokenizer -----------------------------------
def word_tokenize(text):
    """Tokenize into lowercase words, strip punctuation."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]

# ---------- Clean & Normalize Text --------------------------
def clean_text(text):
    """Remove noise: headers, URLs, extra whitespace."""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove citation markers like [1], [2,3], (Smith et al., 2020)
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    text = re.sub(r'\([A-Z][a-z]+\s+et\s+al\.,?\s+\d{4}\)', '', text)
    text = re.sub(r'\([A-Z][a-z]+,?\s+\d{4}\)', '', text)
    # Remove page numbers
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Remove lines that are just numbers or very short
    lines = [l for l in text.split('\n') if len(l.strip()) > 10 or l.strip() == '']
    return '\n'.join(lines).strip()

# ---------- Section Detector --------------------------------
def detect_sections(text):
    """Detect section boundaries and return dict of {section: content}."""
    sections = {}
    lines = text.split('\n')
    current_section = "preamble"
    current_content = []

    for line in lines:
        stripped = line.strip().lower()
        stripped_alpha = re.sub(r'[\d\.\s]', '', stripped)

        if stripped_alpha in SECTION_HEADERS or stripped in SECTION_HEADERS:
            # Save previous section
            if current_content:
                sections[current_section] = ' '.join(current_content).strip()
            current_section = stripped.strip()
            current_content = []
        else:
            current_content.append(line.strip())

    # Save last section
    if current_content:
        sections[current_section] = ' '.join(current_content).strip()

    return sections

# ---------- Remove Stop Words --------------------------------
def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

# ---------- Term Frequency -----------------------------------
def compute_tf(tokens):
    """Compute normalized term frequency."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}

# ---------- Word Frequency Score ----------------------------
def word_freq_scores(text):
    """Score words by frequency (excluding stopwords)."""
    tokens = remove_stopwords(word_tokenize(text))
    tf = compute_tf(tokens)
    max_freq = max(tf.values()) if tf else 1
    return {w: f / max_freq for w, f in tf.items()}

# ---------- Extract Metadata Heuristics ---------------------
def extract_metadata_heuristic(text):
    """Extract title, authors, year using regex heuristics."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    metadata = {
        "title": "",
        "authors": [],
        "year": "",
        "doi": "",
        "keywords": [],
        "venue": ""
    }

    # Title: usually first non-empty long line
    for line in lines[:10]:
        if 15 < len(line) < 200 and not line.startswith(('http', 'doi', 'arXiv')):
            metadata["title"] = line
            break

    # Year: look for 4-digit year pattern 19xx or 20xx
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', text[:3000])
    if years:
        metadata["year"] = max(years)

    # DOI
    doi_match = re.search(r'10\.\d{4,}/\S+', text[:3000])
    if doi_match:
        metadata["doi"] = doi_match.group()

    # Keywords section
    kw_match = re.search(r'[Kk]eywords?\s*[:\-—]\s*(.+)', text[:5000])
    if kw_match:
        kw_line = kw_match.group(1)
        keywords = re.split(r'[,;·•|]', kw_line)
        metadata["keywords"] = [k.strip() for k in keywords if 2 < len(k.strip()) < 50][:10]

    # Authors: lines with comma-separated names near top, before abstract
    author_section = text[:2000]
    # Look for lines with multiple capitalized words (author names)
    for line in lines[1:15]:
        if re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', line) and len(line) < 200:
            if not any(w in line.lower() for w in ['university', 'department', 'abstract', 'institute']):
                names = re.split(r'[,;]|\band\b', line)
                candidates = [n.strip() for n in names if re.search(r'[A-Z][a-z]+', n) and len(n.strip()) < 50]
                if candidates:
                    metadata["authors"] = candidates[:10]
                    break

    return metadata

# ---------- Sentence Position Score -------------------------
def position_scores(sentences):
    """Score sentences by their position (intro & conclusion score higher)."""
    n = len(sentences)
    if n == 0:
        return {}
    scores = {}
    for i, s in enumerate(sentences):
        pos_ratio = i / n
        if pos_ratio < 0.1:       # First 10% — high value
            scores[s] = 1.0
        elif pos_ratio < 0.2:
            scores[s] = 0.8
        elif pos_ratio > 0.85:    # Last 15% — conclusion area
            scores[s] = 0.9
        elif pos_ratio > 0.75:
            scores[s] = 0.7
        else:
            scores[s] = 0.3 + 0.4 * math.exp(-3 * abs(pos_ratio - 0.5))
    return scores
