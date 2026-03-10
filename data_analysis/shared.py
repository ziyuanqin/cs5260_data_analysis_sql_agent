import io
import base64
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")

# LangGraph's MemorySaver uses msgpack, which cannot serialise
# pd.DataFrame.  We keep DataFrames here keyed by table name and
# only store the string key inside the graph state.
_df_store: dict = {}

# to resolve memory error
def _to_serializable(obj):
    """Recursively convert numpy/pandas types to native Python for msgpack."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if v != v or v == float("inf") or v == float("-inf"):
            return None          # msgpack cannot encode inf/nan
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def _llm_text(resp) -> str:
    """
    Normalise LLM response content to a plain string.
    - Anthropic / OpenAI: resp.content is already a str
    - Gemini (langchain-google-genai): resp.content is a list of content blocks
    """
    content = resp.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Each block is either a dict {"type":"text","text":"..."} 
        # or an object with a .text attribute
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)
    return str(content)


# ── Dark-mode colour palette ──────────────────────────────────
BG      = "#0f172a"
CARD    = "#1e293b"
ACCENT  = "#6366f1"
WARN    = "#f59e0b"
DANGER  = "#ef4444"
SUCCESS = "#22c55e"
TEXT    = "#e2e8f0"
MUTED   = "#94a3b8"


# ── Matplotlib helpers ────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    """Render a matplotlib figure to a base64 PNG string and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=BG)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _dark_fig(figsize=(9, 5)):
    """Single-panel dark-themed figure. Returns (fig, ax)."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    return fig, ax


def _dark_fig_multi(nrows, ncols, figsize):
    """Multi-panel dark figure. Returns (fig, flat axes array)."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=BG)
    axes_flat = np.array(axes).flatten()
    for ax in axes_flat:
        ax.set_facecolor(CARD)
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")
        ax.tick_params(colors=MUTED)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)
    return fig, axes_flat
