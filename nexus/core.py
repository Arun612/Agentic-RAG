"""
NEXUS Core — Message Bus + Orchestrator
The orchestrator coordinates the full 10-step query lifecycle.
"""

import time
import logging
from collections import defaultdict

from nexus.config import (
    NexusConfig, LLMClient, console,
    print_agent_action, print_confidence_card, print_sources,
    print_query_plan, print_answer, print_memory_update,
    print_parliament_verdict,
)
from nexus.agents import (
    PlanningAgent, VectorRetriever, KeywordRetriever, WebRetriever,
    reciprocal_rank_fusion, apply_diversity_penalty,
    AdvocateAgent, ProsecutorAgent, JudgeAgent,
    SynthesisAgent, ConfidenceAgent, ToolSmithAgent,
)
from nexus.memory import EpisodicMemory, Episode, SemanticMemory, ProceduralMemory

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Message Bus — inter-agent communication hub
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MessageBus:
    """Central communication hub for agent messaging and tracing."""

    def __init__(self):
        self._message_log = []
        self._trace_counter = 0

    def generate_trace_id(self):
        """Generate a unique trace ID for each query lifecycle."""
        self._trace_counter += 1
        return f"trace_{int(time.time())}_{self._trace_counter}"

    def log_message(self, sender, receiver, content):
        """Log a message between agents."""
        self._message_log.append({
            "sender": sender, "receiver": receiver,
            "content": str(content)[:200], "timestamp": time.time(),
        })

    def get_stats(self):
        return {"total_messages": len(self._message_log)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Orchestrator — the brain that coordinates all agents
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Orchestrator:
    """
    Coordinates the full NEXUS query lifecycle (10 steps):
    1. Check memory → 2. Get strategy → 3. Plan sub-queries (GoTQD)
    4. Retrieve (SPRF) → 5. Verify (Parliament) → 6. Tool Smith
    7. Synthesize → 8. Score confidence → 9. Display → 10. Update memory
    """

    def __init__(self, config: NexusConfig):
        self.config = config
        self.llm_client = LLMClient(config.llm)
        self.message_bus = MessageBus()

        # Initialize all agents
        self.planner = PlanningAgent(self.llm_client)
        self.vector_retriever = VectorRetriever(
            self.llm_client, config.retrieval, config.embedding, config.system.vectorstore_path,
        )
        self.keyword_retriever = KeywordRetriever(self.llm_client, config.retrieval)
        self.web_retriever = WebRetriever(self.llm_client, config.retrieval)
        self.advocate = AdvocateAgent(self.llm_client)
        self.prosecutor = ProsecutorAgent(self.llm_client)
        self.judge = JudgeAgent(self.llm_client)
        self.synthesizer = SynthesisAgent(self.llm_client)
        self.confidence_agent = ConfidenceAgent(self.llm_client)
        self.tool_smith = ToolSmithAgent(self.llm_client)

        # Initialize memory systems
        self.episodic = EpisodicMemory(config.memory.episodic_path, config.memory.max_episodic_entries)
        self.semantic = SemanticMemory(config.memory.semantic_path)
        self.procedural = ProceduralMemory(config.memory.procedural_path)

    def execute(self, query):
        """Execute the full 10-step NEXUS pipeline."""
        start_time = time.time()
        trace_id = self.message_bus.generate_trace_id()

        console.print(f"\n  🔍 [bold]Processing query:[/bold] {query}\n")

        # ─── Step 1: Check Memory ───
        print_agent_action("Orchestrator", "Checking memory for similar past queries...")
        similar_episodes = self.episodic.find_similar(query)
        memory_context = self.semantic.get_context_for_query(query)

        if similar_episodes:
            best = similar_episodes[0]
            print_agent_action("Orchestrator",
                f"Found similar past query (confidence: {best.confidence_score:.0%})",
                f"Past query: \"{best.query[:60]}...\"")

        # ─── Step 2: Get Strategy from Procedural Memory ───
        plan = self.planner.execute(query, memory_context)
        query_type = plan.get("query_type", "factual")
        complexity = plan.get("complexity", "simple")

        recommendation = self.procedural.recommend_strategy(query_type)
        print_agent_action("Orchestrator",
            f"Query classified: {query_type} ({complexity})",
            f"Recommended strategy: {recommendation.get('strategy', 'direct')}")

        # ─── Step 3: Plan Sub-queries (GoTQD — Concept #4) ───
        if plan.get("needs_decomposition", False):
            print_agent_action("Planner", "Decomposing query into sub-queries (GoTQD)...")
            print_query_plan(plan)
        else:
            print_agent_action("Planner", "Simple query — no decomposition needed")

        sub_queries = plan.get("sub_queries", [{"id": "Q1", "query": query, "depends_on": [], "strategy": "vector+web"}])

        # ─── Step 4: Retrieve (SPRF — Concept #3) ───
        print_agent_action("Orchestrator", "Launching parallel retrieval (SPRF)...")
        all_documents = self._execute_retrieval(sub_queries)

        if not all_documents:
            print_agent_action("Orchestrator", "⚠️  No documents — using LLM knowledge only")
            answer = self.llm_client.chat_with_system(
                "Answer the question to the best of your knowledge. Be clear about uncertainty.", query)
            return {"answer": answer, "confidence": {"overall": 0.4}, "sources": [],
                    "trace_id": trace_id, "duration_ms": (time.time() - start_time) * 1000}

        console.print(f"  📄 Retrieved [bold]{len(all_documents)}[/bold] documents total\n")

        # ─── Step 5: Verification Parliament (AVP — Concept #2) ───
        print_agent_action("Orchestrator", "Convening Verification Parliament...")
        verdict, all_documents = self._run_parliament(query, all_documents)
        print_parliament_verdict(verdict)

        # Re-retrieve if REJECTED
        if verdict.get("decision") == "REJECT" and verdict.get("refined_query"):
            console.print("\n  🔄 [yellow]Re-retrieving with refined query...[/yellow]")
            refined_sq = [{"id": "R1", "query": verdict["refined_query"], "depends_on": [], "strategy": "vector+web"}]
            all_documents = self._execute_retrieval(refined_sq)
            if all_documents:
                verdict, all_documents = self._run_parliament(verdict["refined_query"], all_documents)

        # ─── Step 6: Tool Smith (SETS — Concept #7) ───
        print_agent_action("ToolSmith", "Checking if computation is needed...")
        tool_result = self.tool_smith.execute(query, str(all_documents[:2])[:500])
        computation_results = tool_result.get("results", [])
        if tool_result.get("status") == "tool_created":
            print_agent_action("ToolSmith", f"Created new tool: {tool_result.get('tool', 'unknown')}")

        # ─── Step 7: Synthesis ───
        print_agent_action("Synthesizer", "Generating comprehensive answer...")
        synthesis = self.synthesizer.execute(
            query=query,
            documents=all_documents[:self.config.retrieval.max_results],
            computation_results=computation_results,
            memory_context=memory_context,
        )
        answer = synthesis.get("answer", "I was unable to generate an answer.")

        # ─── Step 8: Confidence Scoring (EUQ — Concept #6) ───
        print_agent_action("Confidence", "Scoring answer confidence...")
        confidence = self.confidence_agent.execute(
            query=query, answer=answer,
            sources=synthesis.get("sources", []),
            parliament_scores=verdict.get("scores", {}),
        )

        # ─── Step 9: Display Results ───
        print_answer(answer)
        print_confidence_card(confidence)
        print_sources(synthesis.get("sources", []))

        # ─── Step 10: Update Memory ───
        duration_ms = (time.time() - start_time) * 1000
        self._update_memory(
            query=query, query_type=query_type, complexity=complexity,
            strategy=recommendation.get("strategy", "direct"),
            retrieval_methods=recommendation.get("retrieval_methods", "vector"),
            verdict=verdict, confidence=confidence, answer=answer,
            duration_ms=duration_ms, trace_id=trace_id,
        )
        print_memory_update()

        console.print(f"\n  ⏱️  [dim]Total time: {duration_ms:.0f}ms | "
                      f"Tokens: {self.llm_client.total_tokens_used}[/dim]\n")

        return {
            "answer": answer, "confidence": confidence,
            "sources": synthesis.get("sources", []),
            "verdict": verdict.get("decision"),
            "trace_id": trace_id, "duration_ms": duration_ms,
        }

    def _execute_retrieval(self, sub_queries):
        """Run vector + keyword + web retrieval and fuse with RRF."""
        all_results = {"vector": [], "keyword": [], "web": []}

        for sq in sub_queries:
            strategy = sq.get("strategy", "vector+web")
            sq_query = sq.get("query", "")

            print_agent_action("VectorRetriever", f"Searching: \"{sq_query[:50]}...\"")
            vec_result = self.vector_retriever.execute(sq_query)
            all_results["vector"].extend(vec_result.get("documents", []))

            if "keyword" in strategy:
                print_agent_action("KeywordRetriever", f"Searching: \"{sq_query[:50]}...\"")
                kw_result = self.keyword_retriever.execute(sq_query)
                all_results["keyword"].extend(kw_result.get("documents", []))

            if "web" in strategy and self.config.retrieval.enable_web_search:
                print_agent_action("WebRetriever", f"Searching: \"{sq_query[:50]}...\"")
                web_result = self.web_retriever.execute(sq_query)
                all_results["web"].extend(web_result.get("documents", []))

        # Fuse using Reciprocal Rank Fusion
        result_lists = [v for v in all_results.values() if v]
        if not result_lists:
            return []

        fused = reciprocal_rank_fusion(result_lists)
        return apply_diversity_penalty(fused)

    def _run_parliament(self, query, documents):
        """Run the Adversarial Verification Parliament debate."""
        top_docs = documents[:self.config.retrieval.max_results]

        print_agent_action("Advocate", "Arguing FOR document relevance...")
        advocate_result = self.advocate.execute(query, top_docs)

        print_agent_action("Prosecutor", "Challenging document quality...")
        prosecutor_result = self.prosecutor.execute(query, top_docs)

        print_agent_action("Judge", "Rendering verdict...")
        verdict = self.judge.execute(query, advocate_result, prosecutor_result, top_docs)

        # Filter rejected documents if PARTIAL
        if verdict.get("decision") == "PARTIAL":
            rejected_ids = set(verdict.get("rejected_doc_ids", []))
            if rejected_ids:
                documents = [d for d in documents if d.get("id") not in rejected_ids]

        return verdict, documents

    def _update_memory(self, query, query_type, complexity, strategy,
                       retrieval_methods, verdict, confidence, answer,
                       duration_ms, trace_id):
        """Update all three memory systems after a query completes."""
        # Episodic — record what happened
        episode = Episode(
            query=query, query_type=query_type, complexity=complexity,
            strategy_used=strategy,
            retrieval_methods=retrieval_methods.split("+"),
            parliament_verdict=verdict.get("decision", "UNKNOWN"),
            confidence_score=confidence.get("overall", 0.0),
            answer_summary=answer[:200], duration_ms=duration_ms, trace_id=trace_id,
        )
        self.episodic.record(episode)

        # Procedural — learn from outcomes (MLRO)
        self.procedural.record_outcome(
            query_type=query_type, strategy=strategy,
            retrieval_methods=retrieval_methods,
            success=verdict.get("decision") == "ACCEPT",
            confidence=confidence.get("overall", 0.0),
            duration_ms=duration_ms, tokens_used=self.llm_client.total_tokens_used,
        )

        # Semantic — extract entities from answer into knowledge graph
        try:
            extraction = self.llm_client.chat_json(
                system_prompt="Extract key entities and relationships from this text. "
                "Respond in JSON: {\"entities\": [{\"name\": \"x\", \"type\": \"concept\"}], "
                "\"relations\": [{\"source\": \"x\", \"target\": \"y\", \"type\": \"related_to\"}]}",
                user_message=answer[:1000],
            )
            for ent in extraction.get("entities", [])[:10]:
                self.semantic.add_entity(ent["name"], ent.get("type", "concept"))
            for rel in extraction.get("relations", [])[:10]:
                self.semantic.add_relation(rel["source"], rel["target"], rel.get("type", "related_to"))
        except Exception as e:
            logger.debug(f"Semantic extraction failed (non-critical): {e}")

    def ingest_documents(self, documents):
        """Ingest documents into both vector and keyword indexes."""
        self.vector_retriever.ingest_documents(documents)
        self.keyword_retriever.ingest_documents(documents)
        console.print(f"  ✅ Ingested {len(documents)} documents into NEXUS")

    def get_system_stats(self):
        """Return comprehensive system statistics."""
        return {
            "llm": self.llm_client.get_stats(),
            "message_bus": self.message_bus.get_stats(),
            "memory": {
                "episodic_entries": len(self.episodic.episodes),
                "semantic_entities": len(self.semantic.entities),
                "procedural_records": len(self.procedural.records),
            },
            "vectorstore_docs": self.vector_retriever.get_document_count(),
            "performance_matrix": self.procedural.get_performance_matrix(),
        }
