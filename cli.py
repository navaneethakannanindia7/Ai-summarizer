#!/usr/bin/env python3
# =============================================================
# cli.py  —  Step 6: Command-Line Interface
# Usage: python cli.py paper.pdf [--method textrank|lsa|freq]
#        python cli.py --text "paste abstract here"
# =============================================================

import argparse
import sys
import os
import json
import re

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from preprocessor import clean_text, extract_metadata_heuristic, detect_sections
from summarizer import textrank_summarize, lsa_summarize, frequency_summarize, extract_keywords
from topic_model import lda_topics, nmf_topics, classify_domain, extract_contributions, extract_limitations
from reviewer import score_paper

# ── ANSI Colors ───────────────────────────────────────────────
class C:
    HEADER  = '\033[95m'
    BLUE    = '\033[94m'
    CYAN    = '\033[96m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    RED     = '\033[91m'
    BOLD    = '\033[1m'
    DIM     = '\033[2m'
    RESET   = '\033[0m'

def hr(char="─", width=60, color=C.DIM):
    print(f"{color}{char * width}{C.RESET}")

def header(title):
    hr("═")
    print(f"{C.BOLD}{C.CYAN}  {title}{C.RESET}")
    hr("═")

def section(title):
    print(f"\n{C.BOLD}{C.YELLOW}▶ {title}{C.RESET}")
    hr("─", 50)

def score_bar(score, max_score=10, width=20):
    filled = int((score / max_score) * width)
    bar = "█" * filled + "░" * (width - filled)
    color = C.GREEN if score >= 7 else C.YELLOW if score >= 5 else C.RED
    return f"{color}{bar}{C.RESET} {score:.1f}/{max_score}"


def load_text(args):
    """Load paper text from file or stdin."""
    if args.file:
        path = args.file
        if not os.path.exists(path):
            print(f"{C.RED}Error: File not found: {path}{C.RESET}")
            sys.exit(1)

        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            try:
                import fitz
                doc = fitz.open(path)
                text = "".join(page.get_text() for page in doc)
                doc.close()
                return text
            except ImportError:
                try:
                    from pdfminer.high_level import extract_text
                    return extract_text(path)
                except ImportError:
                    print(f"{C.RED}No PDF library found. Install: pip install PyMuPDF{C.RESET}")
                    sys.exit(1)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    elif args.text:
        return args.text

    elif not sys.stdin.isatty():
        return sys.stdin.read()

    else:
        print(f"{C.RED}Provide a file (--file), text (--text), or pipe via stdin{C.RESET}")
        sys.exit(1)


def run_analysis(text, method="textrank", num_sentences=6, output_json=False):
    """Run full analysis pipeline and display results."""
    text = clean_text(text)

    if len(text) < 100:
        print(f"{C.RED}Error: Text too short (need at least 100 characters){C.RESET}")
        sys.exit(1)

    if output_json:
        # JSON output mode
        metadata = extract_metadata_heuristic(text)
        sections = detect_sections(text)
        summary, _ = textrank_summarize(text, num_sentences)
        keywords = extract_keywords(text, top_n=10)
        scores = score_paper(text)
        domains = classify_domain(text)
        lda = lda_topics(text, n_topics=4)
        contributions = extract_contributions(text)
        limitations = extract_limitations(text)

        result = {
            "metadata": metadata,
            "summary": summary,
            "keywords": [{"term": k, "score": s} for k, s in keywords],
            "topics_lda": lda,
            "domain": domains,
            "contributions": contributions,
            "limitations": limitations,
            "quality_scores": scores,
        }
        print(json.dumps(result, indent=2))
        return

    # ── Pretty print mode ─────────────────────────────────────

    header("🔬 Research Paper Analyzer — Local ML Edition")

    # Metadata
    section("📋 Paper Metadata")
    meta = extract_metadata_heuristic(text)
    print(f"  {C.BOLD}Title    :{C.RESET} {meta.get('title', 'N/A')}")
    print(f"  {C.BOLD}Authors  :{C.RESET} {', '.join(meta.get('authors', [])) or 'N/A'}")
    print(f"  {C.BOLD}Year     :{C.RESET} {meta.get('year', 'N/A')}")
    print(f"  {C.BOLD}DOI      :{C.RESET} {meta.get('doi', 'N/A')}")
    if meta.get("keywords"):
        print(f"  {C.BOLD}Keywords :{C.RESET} {', '.join(meta['keywords'][:6])}")
    print(f"  {C.BOLD}Words    :{C.RESET} ~{len(text.split()):,}")

    # Domain
    section("🌐 Domain Classification")
    domains = classify_domain(text)
    for domain, pct in domains:
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {C.CYAN}{domain:<35}{C.RESET}  {C.GREEN}{bar}{C.RESET}  {pct:.1f}%")

    # Summary
    section(f"📄 Extractive Summary ({method.upper()})")
    if method == "lsa":
        summary, _ = lsa_summarize(text, num_sentences)
    elif method == "freq":
        summary, _ = frequency_summarize(text, num_sentences)
    else:
        summary, _ = textrank_summarize(text, num_sentences)

    # Wrap summary at 80 chars
    words = summary.split()
    line, lines = [], []
    for w in words:
        line.append(w)
        if len(' '.join(line)) > 80:
            lines.append(' '.join(line[:-1]))
            line = [w]
    if line:
        lines.append(' '.join(line))
    for l in lines:
        print(f"  {l}")

    # Keywords
    section("🔑 Key Terms (TF-IDF)")
    kws = extract_keywords(text, top_n=12)
    kw_line = []
    for term, score in kws[:12]:
        kw_line.append(f"{C.CYAN}{term}{C.RESET}({score:.2f})")
    print("  " + "  •  ".join(kw_line))

    # Topics
    section("🗂  Topics (LDA)")
    topics = lda_topics(text, n_topics=4)
    for t in topics:
        words_str = ", ".join(t["words"][:6])
        print(f"  {C.BOLD}Topic {t['topic_id']}: {t['label']:<25}{C.RESET}  [{words_str}]")

    # Contributions
    contribs = extract_contributions(text)
    if contribs:
        section("💡 Stated Contributions")
        for i, c in enumerate(contribs, 1):
            print(f"  {i}. {c}")

    # Limitations
    limits = extract_limitations(text)
    if limits:
        section("⚠️  Limitations / Future Work")
        for i, l in enumerate(limits, 1):
            print(f"  {i}. {l}")

    # Quality Scores
    section("⭐ Quality Assessment")
    scores = score_paper(text)
    dimensions = ["novelty", "clarity", "methodology", "evidence", "impact", "completeness", "writing"]
    for dim in dimensions:
        s = scores[dim]["score"]
        print(f"  {dim.capitalize():<16} {score_bar(s)}")
        print(f"  {C.DIM}{'':16} {scores[dim]['justification'][:75]}...{C.RESET}" if len(scores[dim]['justification']) > 75 else f"  {C.DIM}{'':16} {scores[dim]['justification']}{C.RESET}")

    hr("═")
    rec = scores["recommendation"]
    color = C.GREEN if rec["decision"].startswith("Accept") else (C.YELLOW if "Minor" in rec["decision"] else C.RED)
    print(f"\n  {C.BOLD}Overall Score: {score_bar(scores['overall'])}  →  {color}{rec['decision']}{C.RESET}")
    print(f"  {C.BOLD}Reading Difficulty: {scores['reading_difficulty']}{C.RESET}")
    print(f"  {C.BOLD}Paper Type: {scores['paper_type']}{C.RESET}\n")

    print(f"\n  {C.BOLD}Strengths:{C.RESET}")
    for s in scores["strengths"]:
        print(f"  {C.GREEN}✓{C.RESET} {s}")

    print(f"\n  {C.BOLD}Weaknesses:{C.RESET}")
    for w in scores["weaknesses"]:
        print(f"  {C.RED}✗{C.RESET} {w}")

    hr("═")
    print(f"\n  {C.DIM}Analysis powered by: TextRank · LSA · LDA · NMF · TF-IDF{C.RESET}\n")


# =============================================================
# MAIN
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="📄 Research Paper Analyzer — Local ML (no API needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --file paper.pdf
  python cli.py --file paper.txt --method lsa --sentences 8
  python cli.py --text "Abstract: We propose..."
  python cli.py --file paper.pdf --json > result.json
  cat paper.txt | python cli.py
        """
    )
    parser.add_argument("--file", "-f", help="Path to PDF or TXT file")
    parser.add_argument("--text", "-t", help="Paper text (abstract/full text)")
    parser.add_argument("--method", "-m", choices=["textrank", "lsa", "freq"],
                        default="textrank", help="Summarization method (default: textrank)")
    parser.add_argument("--sentences", "-s", type=int, default=6,
                        help="Number of summary sentences (default: 6)")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of pretty print")

    args = parser.parse_args()
    text = load_text(args)
    run_analysis(text, method=args.method, num_sentences=args.sentences, output_json=args.json)


if __name__ == "__main__":
    main()
