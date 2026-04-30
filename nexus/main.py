"""
NEXUS Main Entry Point — Interactive CLI Interface
Run with: python -m nexus.main
"""

import os
import sys
import logging
import hashlib

from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box

from nexus.config import load_config, print_banner, console
from nexus.core import Orchestrator


def setup_logging(level="INFO"):
    """Configure logging for the NEXUS system."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler("data/nexus.log", mode="a"),
            logging.StreamHandler(sys.stdout) if level == "DEBUG" else logging.NullHandler(),
        ],
    )


def ingest_file(orchestrator, filepath):
    """Ingest a text or PDF file into the NEXUS knowledge base."""
    if not os.path.exists(filepath):
        console.print(f"  ❌ File not found: {filepath}", style="red")
        return

    console.print(f"\n  📥 Ingesting: {filepath}...")

    content = ""
    if filepath.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            content = "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            console.print("  ❌ pypdf not installed. Run: pip install pypdf", style="red")
            return
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

    if not content.strip():
        console.print("  ❌ File is empty or could not be read.", style="red")
        return

    # Chunk the content
    chunk_size = 512
    overlap = 50
    chunks = []
    for i in range(0, len(content), chunk_size - overlap):
        chunk = content[i:i + chunk_size]
        if chunk.strip():
            chunk_id = hashlib.md5(f"{filepath}_{i}".encode()).hexdigest()[:12]
            chunks.append({
                "id": chunk_id,
                "content": chunk.strip(),
                "metadata": {
                    "source": os.path.basename(filepath),
                    "chunk_index": len(chunks),
                    "title": os.path.basename(filepath),
                },
            })

    orchestrator.ingest_documents(chunks)
    console.print(f"  ✅ Ingested {len(chunks)} chunks from {os.path.basename(filepath)}\n")


def show_stats(orchestrator):
    """Display system statistics."""
    stats = orchestrator.get_system_stats()

    table = Table(title="📊 NEXUS System Statistics", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Metric", style="white")
    table.add_column("Value", style="green")

    table.add_row("LLM", "Total API calls", str(stats["llm"]["total_calls"]))
    table.add_row("LLM", "Total tokens used", f"{stats['llm']['total_tokens']:,}")
    table.add_row("LLM", "Estimated cost", f"${stats['llm']['estimated_cost_usd']:.4f}")
    table.add_section()
    table.add_row("Vector Store", "Documents indexed", str(stats["vectorstore_docs"]))
    table.add_section()
    table.add_row("Episodic Memory", "Past interactions", str(stats["memory"]["episodic_entries"]))
    table.add_row("Semantic Memory", "Knowledge entities", str(stats["memory"]["semantic_entities"]))
    table.add_row("Procedural Memory", "Strategy records", str(stats["memory"]["procedural_records"]))
    table.add_section()
    table.add_row("Message Bus", "Total messages", str(stats["message_bus"]["total_messages"]))

    console.print(table)

    # Performance matrix
    matrix = stats.get("performance_matrix", [])
    if matrix:
        console.print("\n")
        perf_table = Table(title="⚡ Strategy Performance Matrix (MLRO)", box=box.SIMPLE)
        perf_table.add_column("Query Type")
        perf_table.add_column("Strategy")
        perf_table.add_column("Retrieval")
        perf_table.add_column("Attempts")
        perf_table.add_column("Success Rate")
        perf_table.add_column("Avg Confidence")

        for row in matrix:
            perf_table.add_row(
                row["query_type"], row["strategy"], row["retrieval"],
                str(row["attempts"]), row["success_rate"], row["avg_confidence"],
            )
        console.print(perf_table)


def show_help():
    """Display available commands."""
    table = Table(title="🔧 NEXUS Commands", box=box.ROUNDED)
    table.add_column("Command", style="cyan bold")
    table.add_column("Description", style="white")

    commands = [
        ("/ingest <file>", "Ingest a document (txt, pdf) into the knowledge base"),
        ("/stats", "Show system statistics and performance matrix"),
        ("/memory", "Show memory system status"),
        ("/clear", "Clear all memory (fresh start)"),
        ("/help", "Show this help message"),
        ("/quit", "Exit NEXUS"),
    ]
    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console.print(table)


def main():
    """Main entry point for the NEXUS CLI."""
    # Ensure data directories exist
    os.makedirs("data/documents", exist_ok=True)
    os.makedirs("data/vectorstore", exist_ok=True)
    os.makedirs("data/memory", exist_ok=True)

    config = load_config()
    setup_logging(config.system.log_level)

    print_banner()

    # Validate API key
    if not config.llm.api_key or config.llm.api_key == "your-groq-api-key-here":
        console.print("  ⚠️  [bold yellow]Groq API key not set![/bold yellow]")
        console.print("  Edit [cyan].env[/cyan] and set your GROQ_API_KEY\n")
        return

    console.print(f"  ✅ Model: [cyan]{config.llm.model}[/cyan]")
    console.print(f"  ✅ Web Search: [cyan]{'Enabled' if config.retrieval.enable_web_search and config.retrieval.tavily_api_key else 'Disabled'}[/cyan]")
    console.print(f"  ✅ Tool Synthesis: [cyan]{'Enabled' if config.system.enable_tool_synthesis else 'Disabled'}[/cyan]")
    console.print()

    # Initialize the Orchestrator
    with console.status("[bold green]Initializing NEXUS agents...[/bold green]"):
        orchestrator = Orchestrator(config)

    doc_count = orchestrator.vector_retriever.get_document_count()
    console.print(f"  📚 Knowledge base: [cyan]{doc_count}[/cyan] documents indexed")
    console.print(f"  🧠 Memory: [cyan]{len(orchestrator.episodic.episodes)}[/cyan] past interactions\n")

    if doc_count == 0:
        console.print("  💡 [dim]Tip: Use [cyan]/ingest <filepath>[/cyan] to add documents to the knowledge base[/dim]\n")

    show_help()
    console.print()

    # Interactive loop
    while True:
        try:
            query = Prompt.ask("\n  [bold cyan]NEXUS[/bold cyan]")
            query = query.strip()

            if not query:
                continue

            # Handle commands
            if query.startswith("/"):
                cmd_parts = query.split(maxsplit=1)
                cmd = cmd_parts[0].lower()

                if cmd == "/quit" or cmd == "/exit":
                    console.print("\n  👋 [bold]Goodbye![/bold]\n")
                    break
                elif cmd == "/help":
                    show_help()
                elif cmd == "/stats":
                    show_stats(orchestrator)
                elif cmd == "/memory":
                    console.print(f"  📝 Episodic: {len(orchestrator.episodic.episodes)} entries")
                    console.print(f"  🧠 Semantic: {orchestrator.semantic.get_stats()}")
                    console.print(f"  ⚙️  Procedural: {len(orchestrator.procedural.records)} records")
                elif cmd == "/ingest" and len(cmd_parts) > 1:
                    ingest_file(orchestrator, cmd_parts[1].strip())
                elif cmd == "/clear":
                    if Confirm.ask("  Clear all memory?"):
                        orchestrator.episodic.clear()
                        console.print("  ✅ Memory cleared")
                else:
                    console.print(f"  ❓ Unknown command: {cmd}. Type /help for available commands.")
                continue

            # Execute query through the full NEXUS pipeline
            console.print()
            result = orchestrator.execute(query)

            # Ask for feedback (for MLRO learning)
            feedback = Prompt.ask(
                "  [dim]Was this helpful? (👍/👎/skip)[/dim]",
                choices=["👍", "👎", "skip", "y", "n", "s"],
                default="skip",
            )
            if feedback in ("👍", "y"):
                if orchestrator.episodic.episodes:
                    orchestrator.episodic.episodes[-1].user_feedback = "thumbs_up"
                    orchestrator.episodic._save()
                console.print("  [green]Thanks! This improves future answers.[/green]")
            elif feedback in ("👎", "n"):
                if orchestrator.episodic.episodes:
                    orchestrator.episodic.episodes[-1].user_feedback = "thumbs_down"
                    orchestrator.episodic._save()
                console.print("  [yellow]Noted. The system will adjust its strategies.[/yellow]")

        except KeyboardInterrupt:
            console.print("\n\n  👋 [bold]Goodbye![/bold]\n")
            break
        except Exception as e:
            console.print(f"\n  ❌ [red]Error: {e}[/red]")
            logging.getLogger(__name__).exception("Unhandled error")


if __name__ == "__main__":
    main()
