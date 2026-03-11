# 🔬 Research Paper AI Analyzer
### Local ML/DL System — No API Key, No Internet Required

---

## 📁 Project Structure

```
research-ai/
├── backend/
│   ├── preprocessor.py     # Step 1 — Text cleaning, tokenization, metadata
│   ├── summarizer.py       # Step 2 — TextRank, LSA, TF-IDF summarization
│   ├── topic_model.py      # Step 3 — LDA, NMF topic modeling
│   ├── reviewer.py         # Step 4 — Quality scoring (7 dimensions)
│   └── app.py              # Step 5 — Flask REST API server
├── frontend/
│   └── index.html          # Step 7 — Full web UI (single file)
├── cli.py                  # Step 6 — Command-line tool
├── requirements.txt
└── README.md
```

---

## 🧠 ML Models Used

| Module | Algorithm | Purpose |
|--------|-----------|---------|
| `summarizer.py` | **TextRank** (PageRank on sentence graph) | Extractive summarization |
| `summarizer.py` | **LSA** (TF-IDF + TruncatedSVD) | Latent semantic summarization |
| `summarizer.py` | **TF-IDF Frequency** | Baseline word-score summarizer |
| `topic_model.py` | **LDA** (Latent Dirichlet Allocation) | Probabilistic topic discovery |
| `topic_model.py` | **NMF** (Non-negative Matrix Factorization) | Sharp topic extraction |
| `topic_model.py` | **Keyword voting** | Domain classification |
| `reviewer.py` | **Heuristic feature scoring** | 7-dim quality assessment |
| `preprocessor.py` | **Regex + TF-IDF** | Metadata & section extraction |

---

## ⚙️ Setup & Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt

# Or individually:
pip install scikit-learn numpy flask flask-cors PyMuPDF
```

### 2. Start the Backend

```bash
cd backend
python app.py
```
→ Server runs at `http://localhost:5000`

### 3. Open the Frontend

```bash
# Just open in browser:
open frontend/index.html

# Or serve with Python:
cd frontend && python -m http.server 8080
```
→ Open `http://localhost:8080`

---

## 🖥️ CLI Usage

```bash
cd backend

# Analyze a PDF
python cli.py --file paper.pdf

# Analyze a TXT file with LSA summarization
python cli.py --file paper.txt --method lsa --sentences 8

# Analyze pasted text
python cli.py --text "Abstract: We propose a novel method..."

# Pipe from stdin
cat paper.txt | python cli.py

# Export JSON
python cli.py --file paper.pdf --json > output.json
```

---

## 🌐 REST API Reference

All endpoints served at `http://localhost:5000/api`

### Upload a Paper

```bash
# Upload PDF
curl -X POST http://localhost:5000/api/upload \
  -F "file=@paper.pdf"

# Upload text
curl -X POST http://localhost:5000/api/upload \
  -H "Content-Type: application/json" \
  -d '{"text": "Abstract: We propose...", "title": "My Paper"}'
```

Response:
```json
{
  "success": true,
  "paper_id": "a1b2c3d4",
  "filename": "paper.pdf",
  "word_count": 8432
}
```

### Full Analysis

```bash
curl http://localhost:5000/api/analyze/a1b2c3d4
```

Returns: metadata, summaries (TextRank/LSA/Section), keywords, LDA topics, NMF topics, domain classification, contributions, limitations, quality scores.

### Summarize Only

```bash
curl -X POST http://localhost:5000/api/summarize/a1b2c3d4 \
  -H "Content-Type: application/json" \
  -d '{"method": "textrank", "num_sentences": 5}'
```
Methods: `textrank` | `lsa` | `frequency`

### Quality Review

```bash
curl http://localhost:5000/api/review/a1b2c3d4
```

### Topics Only

```bash
curl "http://localhost:5000/api/topics/a1b2c3d4?n=5"
```

### List Papers

```bash
curl http://localhost:5000/api/papers
```

---

## 📊 Quality Scoring (7 Dimensions)

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| **Novelty** | 20% | Novelty/contribution term density, comparison claims |
| **Methodology** | 25% | Rigor terms, method section, ablation, reproducibility |
| **Evidence** | 20% | Figures, tables, equations, citation count |
| **Clarity** | 15% | Sentence length variance, section structure |
| **Impact** | 10% | Real-world application discussion |
| **Completeness** | 5% | Presence of essential sections, word count |
| **Writing** | 5% | Passive voice ratio, sentence length distribution |

**Scoring:**
- **8.0–10.0** → Accept
- **6.5–7.9** → Accept with Minor Revisions
- **5.0–6.4** → Major Revisions
- **< 5.0** → Reject

---

## 🔬 Algorithm Deep Dive

### TextRank Summarization
1. Tokenize text into sentences
2. Build TF-IDF vectors for each sentence
3. Compute cosine similarity matrix → weighted graph
4. Run PageRank iterations (damping=0.85) until convergence
5. Blend scores with positional bias (intro/conclusion get boost)
6. Return top-N sentences in original order

### LSA Summarization
1. TF-IDF vectorize sentences
2. Apply `TruncatedSVD` to get latent semantic dimensions
3. Score sentences by L2 norm across all latent axes
4. Blend with positional bias → return top-N

### LDA Topic Modeling
1. `CountVectorizer` → document-term matrix
2. `LatentDirichletAllocation` (online learning, 20 iterations)
3. Extract top words per topic
4. Auto-label topics from meaningful words

### NMF Topic Modeling
1. `TfidfVectorizer` → TF-IDF matrix
2. `NMF` with NNDSVDA initialization
3. Extract components → sharper, more focused topics than LDA

---

## 🏭 Production Notes

- **Replace** in-memory `PAPERS` dict with SQLite or PostgreSQL
- **Add** authentication middleware
- **Use** Gunicorn for production: `gunicorn -w 4 app:app`
- **Add** Redis caching for analysis results
- **Add** Celery for async background analysis
- **Scale** with Docker: `docker-compose up`

---

## 📝 License

MIT — Free for academic and commercial use.
