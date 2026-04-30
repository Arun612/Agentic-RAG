"""
NEXUS Config & Utilities
Configuration, LLM client, embeddings engine, and terminal formatters.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.tree import Tree
from rich import box

load_dotenv()
logger = logging.getLogger(__name__)
console = Console()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Configuration Dataclasses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class LLMConfig:
    """Groq LLM provider settings."""
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    base_url: str = "https://api.groq.com/openai/v1"
    temperature: float = 0.3
    max_tokens: int = 4096


@dataclass
class EmbeddingConfig:
    """Local sentence-transformers embedding settings."""
    provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "local"))
    local_model: str = "all-MiniLM-L6-v2"


@dataclass
class RetrievalConfig:
    """Retrieval pipeline settings."""
    max_results: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIEVAL_RESULTS", "5")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    enable_web_search: bool = field(default_factory=lambda: os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true")
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class MemoryConfig:
    """Memory storage paths and limits."""
    episodic_path: str = "data/memory/episodic.json"
    semantic_path: str = "data/memory/semantic.json"
    procedural_path: str = "data/memory/procedural.json"
    max_episodic_entries: int = 1000
    consolidation_threshold: int = 50


@dataclass
class SystemConfig:
    """System-wide settings."""
    confidence_threshold: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.6")))
    enable_tool_synthesis: bool = field(default_factory=lambda: os.getenv("ENABLE_TOOL_SYNTHESIS", "true").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    vectorstore_path: str = "data/vectorstore"
    documents_path: str = "data/documents"


@dataclass
class NexusConfig:
    """Master config container."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


def load_config():
    """Load and return the complete NEXUS configuration."""
    return NexusConfig()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LLM Client (Groq via OpenAI-compatible API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LLMClient:
    """Handles all LLM calls through Groq's free API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model = config.model
        self.total_tokens_used = 0
        self.total_calls = 0

    def chat(self, messages, temperature=None, max_tokens=None, tools=None, response_format=None):
        """Send a chat completion request and return structured result."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**kwargs)
            self.total_calls += 1
            if response.usage:
                self.total_tokens_used += response.usage.total_tokens

            message = response.choices[0].message
            result = {
                "content": message.content or "",
                "tool_calls": None,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            }
            if message.tool_calls:
                result["tool_calls"] = [
                    {"id": tc.id, "function_name": tc.function.name,
                     "arguments": json.loads(tc.function.arguments)}
                    for tc in message.tool_calls
                ]
            return result
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def chat_with_system(self, system_prompt, user_message, temperature=None, response_format=None):
        """Quick helper: system + user message → response string."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self.chat(messages, temperature=temperature, response_format=response_format)["content"]

    def chat_json(self, system_prompt, user_message, temperature=None):
        """Get a parsed JSON response from the LLM."""
        content = self.chat_with_system(
            system_prompt, user_message, temperature,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"JSON parse failed: {content[:200]}")
            return {"error": "Failed to parse JSON", "raw": content}

    def get_stats(self):
        return {"total_calls": self.total_calls, "total_tokens": self.total_tokens_used,
                "estimated_cost_usd": 0.0}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Embedding Engine (Local sentence-transformers)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EmbeddingEngine:
    """Generates text embeddings using a local model (all-MiniLM-L6-v2)."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(config.local_model)
        logger.info(f"Loaded embedding model: {config.local_model}")

    def embed(self, text):
        """Embed a single text string → list of floats."""
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts):
        """Embed multiple texts at once → list of lists."""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    @staticmethod
    def cosine_similarity(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Terminal Formatters (Rich CLI output)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AGENT_COLORS = {
    "Orchestrator": "bold red", "Planner": "bold yellow",
    "VectorRetriever": "bold green", "KeywordRetriever": "bold blue",
    "WebRetriever": "bold magenta", "Advocate": "bold green",
    "Prosecutor": "bold red", "Judge": "bold yellow",
    "Synthesizer": "bold cyan", "ToolSmith": "bold magenta",
    "Confidence": "bold blue",
}


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗               ║
║   ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝               ║
║   ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗               ║
║   ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║               ║
║   ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║               ║
║   ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝               ║
║                                                              ║
║   Neuro-Episodic Expert Unified System                       ║
║   Multi-Agent Agentic RAG v1.0                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="bold cyan")


def print_agent_action(name, action, detail=""):
    color = AGENT_COLORS.get(name, "white")
    console.print(f"  🤖 [{color}]{name}[/{color}] → {action}", highlight=False)
    if detail:
        console.print(f"      {detail}", style="dim")


def print_confidence_card(confidence):
    """Display a rich confidence scorecard with bar charts."""
    table = Table(title="🎯 Confidence Card", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Dimension", width=22)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Bar", width=30)

    dims = [
        ("factual_accuracy", "Factual Accuracy", "green"),
        ("source_agreement", "Source Agreement", "blue"),
        ("temporal_relevance", "Temporal Relevance", "yellow"),
        ("coverage", "Coverage", "magenta"),
        ("overall", "OVERALL", "bold cyan"),
    ]
    for key, label, color in dims:
        score = confidence.get(key, 0.0)
        pct = int(score * 100)
        filled = int(score * 20)
        bar = "█" * filled + "░" * (20 - filled)
        if key == "overall":
            table.add_section()
        table.add_row(label, f"{pct}%", Text(bar, style=color))

    console.print(table)

    for flag in confidence.get("flags", []):
        console.print(f"     ⚠️  {flag}", style="yellow")
    for unk in confidence.get("unknowns", []):
        console.print(f"     ❓ {unk}", style="blue")


def print_sources(sources):
    if not sources:
        return
    table = Table(title="📚 Sources", box=box.SIMPLE, header_style="bold")
    table.add_column("#", width=3)
    table.add_column("Source", width=40)
    table.add_column("Type", width=10)
    table.add_column("Relevance", justify="center", width=10)
    for i, s in enumerate(sources, 1):
        table.add_row(str(i), s.get("title", "Unknown")[:40], s.get("type", "doc"), f"{s.get('relevance', 0):.0%}")
    console.print(table)


def print_query_plan(plan):
    tree = Tree("🌳 [bold]Query Decomposition Plan[/bold]")
    for sq in plan.get("sub_queries", []):
        deps = sq.get("depends_on", [])
        dep_str = f" [dim](depends: {', '.join(deps)})[/dim]" if deps else " [green](independent)[/green]"
        node = tree.add(f"[bold]{sq.get('id')}[/bold]: {sq.get('query', '')}{dep_str}")
        node.add(f"[dim]Strategy: {sq.get('strategy', 'auto')}[/dim]")
    console.print(tree)


def print_answer(answer):
    console.print()
    console.print(Panel(
        Markdown(answer), title="[bold green]📋 NEXUS Answer[/bold green]",
        border_style="green", padding=(1, 2),
    ))


def print_parliament_verdict(verdict):
    decision = verdict.get("decision", "UNKNOWN")
    colors = {"ACCEPT": "green", "REJECT": "red", "PARTIAL": "yellow"}
    color = colors.get(decision, "white")
    console.print(f"\n  ⚖️  Parliament Verdict: [{color} bold]{decision}[/{color} bold]")
    if verdict.get("advocate_summary"):
        console.print(f"     ✅ Advocate: {verdict['advocate_summary']}", style="green")
    if verdict.get("prosecutor_summary"):
        console.print(f"     ❌ Prosecutor: {verdict['prosecutor_summary']}", style="red")
    if verdict.get("judge_reasoning"):
        console.print(f"     ⚖️  Judge: {verdict['judge_reasoning']}", style="yellow")


def print_memory_update():
    console.print("\n  💾 [dim]Memory updated: 📝 Episodic | 🧠 Semantic | ⚙️ Procedural[/dim]")
