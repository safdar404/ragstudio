"""Central configuration: limits, defaults and retrieval settings."""
from dataclasses import dataclass

APP_NAME = "Document Studio"
VERSION = "2.0"

# ── Limits ──────────────────────────────────────────────────────────────
MAX_FILES_PER_BATCH = 20
MAX_FILE_MB = 25
MAX_PDF_PAGES = 500
MAX_OCR_PAGES_PER_PDF = 25          # scanned pages sent to the vision model
MAX_LIBRARY_PASSAGES = 6000
MAX_DOCX_UNCOMPRESSED_MB = 80
MAX_CSV_ROWS = 20000
IMAGE_MAX_SIDE = 2000
MAX_CONTEXT_CHARS = 28000           # hard cap on evidence sent per answer

# ── Chunking ────────────────────────────────────────────────────────────
CHUNK_SIZE = 1100                   # characters
CHUNK_OVERLAP = 180                 # characters (sentence-aligned)
EMBED_BATCH = 96

# ── Models ──────────────────────────────────────────────────────────────
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_VISION_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]
TEXT_TYPES = ["txt", "md", "csv"]
SUPPORTED_TYPES = ["pdf", "docx", *IMAGE_TYPES, *TEXT_TYPES]

ANSWER_STYLES = {
    "Balanced": "Answer clearly in one to three short paragraphs.",
    "Concise": "Answer in at most three sentences.",
    "Detailed": "Give a thorough, well-structured answer. Use short headings or lists where they help.",
    "Bullet points": "Answer as a compact bulleted list.",
}

RETRIEVAL_MODES = ["Hybrid", "Semantic", "Keyword"]

STARTER_QUESTIONS = [
    "Summarize the main points of these documents.",
    "What are the key figures, dates and deadlines?",
    "Which risks, limitations or open issues are mentioned?",
    "How do the documents differ from each other?",
]


@dataclass
class RetrievalSettings:
    mode: str = "Hybrid"            # Hybrid | Semantic | Keyword
    top_k: int = 6                  # passages sent to the answer model
    candidates: int = 30            # pool size before diversification/re-ranking
    use_mmr: bool = True
    mmr_lambda: float = 0.7         # 1.0 = pure relevance, lower = more diverse
    neighbors: bool = False         # add adjacent passages as extra context
    min_similarity: float = 0.0     # drop weak semantic matches
    rewrite: bool = True            # resolve follow-up questions using chat history
    multi_query: bool = False       # generate alternative phrasings
    rerank: bool = False            # LLM relevance grading of candidates
    files: tuple = ()               # restrict to these document ids (empty = all)
