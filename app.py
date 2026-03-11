# =============================================================
# app.py  —  Step 5: Flask REST API Server
# Endpoints for upload, analyze, summarize, review, topics
# =============================================================

import os
import re
import json
import uuid
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

# Local ML modules
from preprocessor import clean_text, extract_metadata_heuristic, detect_sections
from summarizer import (
    textrank_summarize, lsa_summarize, frequency_summarize,
    extract_keywords, section_aware_summarize
)
from topic_model import (
    lda_topics, nmf_topics, classify_domain,
    extract_contributions, extract_limitations
)
from reviewer import score_paper

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory paper store (use SQLite/MongoDB in production)
PAPERS = {}

# ── PDF text extraction (try PyMuPDF, fallback to pdfminer) ──

def extract_pdf_text(filepath):
    """Extract text from PDF — try multiple libraries."""
    # Try PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except ImportError:
        pass

    # Try pdfminer
    try:
        from pdfminer.high_level import extract_text
        return extract_text(filepath)
    except ImportError:
        pass

    # Try pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        pass

    raise RuntimeError("No PDF library available. Please install PyMuPDF: pip install PyMuPDF")


# =============================================================
# ROUTES
# =============================================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "papers_loaded": len(PAPERS)})


# ── Upload PDF or paste text ──────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload_paper():
    try:
        text = None
        filename = "pasted_text"

        # File upload
        if "file" in request.files:
            f = request.files["file"]
            if not f.filename.lower().endswith((".pdf", ".txt")):
                return jsonify({"error": "Only PDF and TXT files supported"}), 400

            filename = f.filename
            ext = os.path.splitext(filename)[1].lower()
            save_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}{ext}")
            f.save(save_path)

            if ext == ".pdf":
                text = extract_pdf_text(save_path)
            else:
                with open(save_path, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()

        # Pasted text
        elif request.is_json and request.json.get("text"):
            text = request.json["text"]
            filename = request.json.get("title", "Pasted Paper")

        else:
            return jsonify({"error": "Provide a file or text field"}), 400

        if not text or len(text.strip()) < 200:
            return jsonify({"error": "Could not extract sufficient text (min 200 chars)"}), 400

        text = clean_text(text)
        paper_id = str(uuid.uuid4())[:8]

        PAPERS[paper_id] = {
            "id": paper_id,
            "filename": filename,
            "text": text,
            "word_count": len(text.split()),
            "char_count": len(text),
        }

        return jsonify({
            "success": True,
            "paper_id": paper_id,
            "filename": filename,
            "word_count": PAPERS[paper_id]["word_count"],
            "preview": text[:300] + "..."
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Full analysis pipeline ────────────────────────────────────
@app.route("/api/analyze/<paper_id>", methods=["GET"])
def analyze_paper(paper_id):
    paper = PAPERS.get(paper_id)
    if not paper:
        return jsonify({"error": "Paper not found"}), 404

    try:
        text = paper["text"]

        # 1. Metadata extraction
        metadata = extract_metadata_heuristic(text)
        sections = detect_sections(text)

        # 2. Summarization (TextRank — best quality)
        summary, all_sentences = textrank_summarize(text, num_sentences=6)

        # 3. LSA summary (alternative perspective)
        lsa_summary, _ = lsa_summarize(text, num_sentences=4)

        # 4. Section-aware summary
        section_summary = section_aware_summarize(sections, sentences_per_section=2)

        # 5. Keywords
        keywords = extract_keywords(text, top_n=12)

        # 6. Topics
        lda = lda_topics(text, n_topics=4)
        nmf = nmf_topics(text, n_topics=4)

        # 7. Domain classification
        domains = classify_domain(text)

        # 8. Contributions & Limitations
        contributions = extract_contributions(text)
        limitations = extract_limitations(text)

        # 9. Quality scoring
        scores = score_paper(text, metadata)

        result = {
            "paper_id": paper_id,
            "filename": paper["filename"],
            "word_count": paper["word_count"],
            "metadata": metadata,
            "sections_detected": list(sections.keys()),
            "summary": {
                "textrank": summary,
                "lsa": lsa_summary,
                "section_based": section_summary,
            },
            "keywords": [{"term": k, "score": round(s, 4)} for k, s in keywords],
            "topics": {
                "lda": lda,
                "nmf": nmf,
            },
            "domain_classification": [
                {"domain": d, "confidence_pct": c} for d, c in domains
            ],
            "contributions": contributions,
            "limitations": limitations,
            "quality_scores": scores,
            "sentence_count": len(all_sentences),
        }

        # Cache on paper
        PAPERS[paper_id]["analysis"] = result
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Just summarize ────────────────────────────────────────────
@app.route("/api/summarize/<paper_id>", methods=["POST"])
def summarize(paper_id):
    paper = PAPERS.get(paper_id)
    if not paper:
        return jsonify({"error": "Paper not found"}), 404

    data = request.json or {}
    method = data.get("method", "textrank")      # textrank | lsa | frequency
    num_sentences = int(data.get("num_sentences", 5))

    try:
        text = paper["text"]
        if method == "lsa":
            summary, _ = lsa_summarize(text, num_sentences)
        elif method == "frequency":
            summary, _ = frequency_summarize(text, num_sentences)
        else:
            summary, _ = textrank_summarize(text, num_sentences)

        return jsonify({
            "paper_id": paper_id,
            "method": method,
            "num_sentences": num_sentences,
            "summary": summary
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Just score/review ─────────────────────────────────────────
@app.route("/api/review/<paper_id>", methods=["GET"])
def review(paper_id):
    paper = PAPERS.get(paper_id)
    if not paper:
        return jsonify({"error": "Paper not found"}), 404
    try:
        scores = score_paper(paper["text"])
        return jsonify({"paper_id": paper_id, "review": scores})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Get topics ────────────────────────────────────────────────
@app.route("/api/topics/<paper_id>", methods=["GET"])
def topics(paper_id):
    paper = PAPERS.get(paper_id)
    if not paper:
        return jsonify({"error": "Paper not found"}), 404
    try:
        n = int(request.args.get("n", 5))
        return jsonify({
            "paper_id": paper_id,
            "lda": lda_topics(paper["text"], n_topics=n),
            "nmf": nmf_topics(paper["text"], n_topics=n),
            "domain": classify_domain(paper["text"])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── List all loaded papers ────────────────────────────────────
@app.route("/api/papers", methods=["GET"])
def list_papers():
    return jsonify({
        "papers": [
            {
                "id": p["id"],
                "filename": p["filename"],
                "word_count": p["word_count"],
                "analyzed": "analysis" in p
            }
            for p in PAPERS.values()
        ]
    })


# ── Delete paper ──────────────────────────────────────────────
@app.route("/api/papers/<paper_id>", methods=["DELETE"])
def delete_paper(paper_id):
    if paper_id in PAPERS:
        del PAPERS[paper_id]
        return jsonify({"success": True})
    return jsonify({"error": "Not found"}), 404


# =============================================================
if __name__ == "__main__":
    print("\n🔬 Research Paper AI — Local ML Backend")
    print("=" * 45)
    print("Models: TextRank | LSA | LDA | NMF | TF-IDF")
    print("No internet or API key required!")
    print("=" * 45)
    app.run(debug=True, host="0.0.0.0", port=5000)
