"""
NEXUS Memory Systems — Cognitive Memory Fabric (Concept #1)
Three-tier memory: episodic (what happened), semantic (knowledge graph),
and procedural (strategy optimization via MLRO — Concept #5).
"""

import json
import os
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict

import networkx as nx

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Episodic Memory — records WHAT HAPPENED in each interaction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Episode:
    """A single interaction trace — one complete query lifecycle."""
    query: str
    query_type: str
    complexity: str
    strategy_used: str
    sub_queries: list = field(default_factory=list)
    retrieval_methods: list = field(default_factory=list)
    documents_retrieved: int = 0
    parliament_verdict: str = ""
    confidence_score: float = 0.0
    answer_summary: str = ""
    user_feedback: Optional[str] = None
    duration_ms: float = 0.0
    tokens_used: int = 0
    timestamp: float = field(default_factory=time.time)
    trace_id: str = ""


class EpisodicMemory:
    """
    Stores past interactions so the system can recognize similar situations.
    Uses keyword-based Jaccard similarity for fast in-memory matching.
    """

    def __init__(self, path="data/memory/episodic.json", max_entries=1000):
        self.path = path
        self.max_entries = max_entries
        self.episodes = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.episodes = [Episode(**ep) for ep in json.load(f)]
            except (json.JSONDecodeError, TypeError):
                self.episodes = []

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump([asdict(ep) for ep in self.episodes], f, indent=2)

    def record(self, episode):
        """Store a new episode and enforce max size."""
        self.episodes.append(episode)
        if len(self.episodes) > self.max_entries:
            self.episodes = self.episodes[-self.max_entries:]
        self._save()

    def find_similar(self, query, top_k=3):
        """Find similar past queries using Jaccard similarity on word overlap."""
        if not self.episodes:
            return []
        query_words = set(query.lower().split())
        scored = []
        for ep in self.episodes:
            ep_words = set(ep.query.lower().split())
            intersection = len(query_words & ep_words)
            union = len(query_words | ep_words)
            score = intersection / union if union > 0 else 0
            scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for score, ep in scored[:top_k] if score > 0.15]

    def get_recent(self, n=5):
        return self.episodes[-n:]

    def clear(self):
        self.episodes = []
        self._save()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Semantic Memory — knowledge graph that grows over time
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class KnowledgeEntity:
    """A node in the knowledge graph."""
    name: str
    entity_type: str
    description: str = ""
    confidence: float = 1.0
    source_count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


class SemanticMemory:
    """
    Dynamic knowledge graph using NetworkX.
    Supports multi-hop reasoning (e.g. lithium → batteries → EVs → carbon).
    Entities and relations accumulate from every interaction.
    """

    def __init__(self, path="data/memory/semantic.json"):
        self.path = path
        self.graph = nx.DiGraph()
        self.entities = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            for ent in data.get("entities", []):
                e = KnowledgeEntity(**ent)
                self.entities[e.name] = e
                self.graph.add_node(e.name, **asdict(e))
            for rel in data.get("relations", []):
                self.graph.add_edge(
                    rel["source"], rel["target"],
                    relation_type=rel.get("relation_type", "related_to"),
                    confidence=rel.get("confidence", 1.0),
                    weight=rel.get("weight", 1.0),
                )
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to load semantic memory: {e}")

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = {
            "entities": [asdict(e) for e in self.entities.values()],
            "relations": [
                {"source": u, "target": v,
                 "relation_type": d.get("relation_type", "related_to"),
                 "confidence": d.get("confidence", 1.0),
                 "weight": d.get("weight", 1.0)}
                for u, v, d in self.graph.edges(data=True)
            ],
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def add_entity(self, name, entity_type, description="", confidence=1.0):
        """Add or update an entity. Repeated mentions increase confidence."""
        key = name.lower().strip()
        if key in self.entities:
            e = self.entities[key]
            e.source_count += 1
            e.confidence = (e.confidence + confidence) / 2
            e.last_updated = time.time()
            if description and len(description) > len(e.description):
                e.description = description
        else:
            self.entities[key] = KnowledgeEntity(
                name=key, entity_type=entity_type,
                description=description, confidence=confidence,
            )
            self.graph.add_node(key, **asdict(self.entities[key]))
        self._save()

    def add_relation(self, source, target, relation_type, confidence=1.0):
        """Add or strengthen a relationship between two entities."""
        src, tgt = source.lower().strip(), target.lower().strip()
        # Auto-create entities if missing
        if src not in self.entities:
            self.add_entity(src, "concept")
        if tgt not in self.entities:
            self.add_entity(tgt, "concept")

        if self.graph.has_edge(src, tgt):
            edge = self.graph[src][tgt]
            edge["weight"] = edge.get("weight", 1.0) + 0.5
            edge["confidence"] = (edge.get("confidence", 1.0) + confidence) / 2
        else:
            self.graph.add_edge(src, tgt, relation_type=relation_type,
                                confidence=confidence, weight=1.0)
        self._save()

    def get_context_for_query(self, query):
        """Pull relevant knowledge from the graph based on query keywords."""
        query_words = set(query.lower().split())
        context_parts = []

        for entity_name, entity in self.entities.items():
            entity_words = set(entity_name.split())
            if entity_words & query_words:
                info = f"Known entity: {entity_name}"
                if entity.description:
                    info += f" — {entity.description}"
                # Add connected relations
                for _, target, data in self.graph.out_edges(entity_name, data=True):
                    info += f"\n  → {data.get('relation_type', 'related_to')}: {target}"
                for source, _, data in self.graph.in_edges(entity_name, data=True):
                    info += f"\n  ← {data.get('relation_type', 'related_to')}: {source}"
                context_parts.append(info)

        return "\n\n".join(context_parts) if context_parts else ""

    def find_path(self, source, target, max_hops=3):
        """Multi-hop reasoning: find shortest path between entities."""
        try:
            path = nx.shortest_path(self.graph, source.lower().strip(), target.lower().strip())
            return path if len(path) <= max_hops + 1 else []
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_stats(self):
        return {
            "entities": len(self.entities),
            "relations": self.graph.number_of_edges(),
            "most_connected": sorted(
                [(n, self.graph.degree(n)) for n in self.graph.nodes()],
                key=lambda x: x[1], reverse=True,
            )[:5],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Procedural Memory — Meta-Learning Retrieval Optimizer (MLRO)
#  Concept #5: learns which strategies work best per query type
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class StrategyRecord:
    """Performance record for a strategy + query type combination."""
    query_type: str
    strategy: str
    retrieval_methods: str
    attempts: int = 0
    successes: int = 0
    avg_confidence: float = 0.0
    avg_duration_ms: float = 0.0
    avg_tokens: float = 0.0
    last_used: float = field(default_factory=time.time)

    @property
    def success_rate(self):
        return self.successes / self.attempts if self.attempts > 0 else 0.0


class ProceduralMemory:
    """
    Tracks which retrieval strategies work best for each query type.
    Uses weighted scoring: 50% success rate + 30% confidence + 20% efficiency.
    """

    def __init__(self, path="data/memory/procedural.json"):
        self.path = path
        self.records = {}
        self._load()

    def _make_key(self, query_type, strategy, retrieval_methods):
        return f"{query_type}::{strategy}::{retrieval_methods}"

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                for key, rec in data.items():
                    self.records[key] = StrategyRecord(**rec)
            except (json.JSONDecodeError, TypeError):
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({k: asdict(v) for k, v in self.records.items()}, f, indent=2)

    def record_outcome(self, query_type, strategy, retrieval_methods,
                       success, confidence, duration_ms, tokens_used):
        """Record the result of a strategy execution for learning."""
        key = self._make_key(query_type, strategy, retrieval_methods)
        if key in self.records:
            rec = self.records[key]
            rec.attempts += 1
            if success:
                rec.successes += 1
            # Running averages
            n = rec.attempts
            rec.avg_confidence = ((rec.avg_confidence * (n - 1) + confidence) / n)
            rec.avg_duration_ms = ((rec.avg_duration_ms * (n - 1) + duration_ms) / n)
            rec.avg_tokens = ((rec.avg_tokens * (n - 1) + tokens_used) / n)
            rec.last_used = time.time()
        else:
            self.records[key] = StrategyRecord(
                query_type=query_type, strategy=strategy,
                retrieval_methods=retrieval_methods,
                attempts=1, successes=1 if success else 0,
                avg_confidence=confidence, avg_duration_ms=duration_ms,
                avg_tokens=float(tokens_used),
            )
        self._save()

    def recommend_strategy(self, query_type):
        """Pick the best strategy for a query type based on past performance."""
        candidates = [(k, r) for k, r in self.records.items()
                      if r.query_type == query_type and r.attempts >= 2]

        if not candidates:
            # Default strategies when we have no data yet
            defaults = {
                "factual": {"strategy": "direct", "retrieval_methods": "vector+web"},
                "comparison": {"strategy": "gotqd", "retrieval_methods": "vector+web"},
                "multi-hop": {"strategy": "gotqd", "retrieval_methods": "vector+keyword"},
                "temporal": {"strategy": "direct", "retrieval_methods": "web+vector"},
                "creative": {"strategy": "direct", "retrieval_methods": "vector"},
            }
            return defaults.get(query_type, {"strategy": "direct", "retrieval_methods": "vector+web"})

        # Weighted scoring: success_rate * 0.5 + confidence * 0.3 + efficiency * 0.2
        max_tokens = max(r.avg_tokens for _, r in candidates) or 1
        scored = []
        for key, rec in candidates:
            efficiency = 1.0 - (rec.avg_tokens / max_tokens)
            score = 0.5 * rec.success_rate + 0.3 * rec.avg_confidence + 0.2 * efficiency
            scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        return {"strategy": best.strategy, "retrieval_methods": best.retrieval_methods,
                "confidence": round(scored[0][0], 3), "based_on_attempts": best.attempts}

    def get_performance_matrix(self):
        """Return full performance data for the /stats display."""
        return [
            {"query_type": r.query_type, "strategy": r.strategy,
             "retrieval": r.retrieval_methods, "attempts": r.attempts,
             "success_rate": f"{r.success_rate:.0%}",
             "avg_confidence": f"{r.avg_confidence:.2f}"}
            for _, r in sorted(self.records.items())
        ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Memory Consolidator — compresses episodic → semantic
#  Inspired by sleep consolidation in neuroscience.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MemoryConsolidator:
    """Reviews recent episodes and extracts entities/relations into the knowledge graph."""

    def __init__(self, episodic, semantic, procedural, llm_client):
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural
        self.llm = llm_client

    def consolidate(self, batch_size=20):
        """Run a consolidation cycle over recent episodes."""
        recent = self.episodic.get_recent(batch_size)
        if not recent:
            return

        combined = "\n\n".join([
            f"Query: {ep.query}\nAnswer: {ep.answer_summary}"
            for ep in recent if ep.answer_summary
        ])
        if not combined:
            return

        try:
            extraction = self.llm.chat_json(
                system_prompt=(
                    "Extract key entities and relationships from these Q&A pairs. "
                    "Respond in JSON: {\"entities\": [{\"name\": \"x\", \"type\": \"concept\"}], "
                    "\"relations\": [{\"source\": \"x\", \"target\": \"y\", \"type\": \"related_to\"}]}"
                ),
                user_message=combined[:3000],
            )
            for ent in extraction.get("entities", [])[:20]:
                self.semantic.add_entity(ent["name"], ent.get("type", "concept"))
            for rel in extraction.get("relations", [])[:20]:
                self.semantic.add_relation(rel["source"], rel["target"], rel.get("type", "related_to"))
            logger.info("Memory consolidation complete")
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")

    def should_consolidate(self, threshold=50):
        return len(self.episodic.episodes) % threshold == 0 and len(self.episodic.episodes) > 0
