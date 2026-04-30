"""
NEXUS Agents — All specialized agents in one module.
Includes: planner, retrievers (vector/keyword/web), fusion (RRF),
verification parliament (advocate/prosecutor/judge), synthesis,
confidence scoring, and tool smith.
"""

import os
import json
import logging
from typing import Optional
from collections import defaultdict

import chromadb
from rank_bm25 import BM25Okapi

from nexus.config import LLMClient, EmbeddingEngine, EmbeddingConfig, RetrievalConfig

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Base Agent — simple parent class for all agents
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BaseAgent:
    """All agents inherit from this. Provides LLM helper methods."""

    def __init__(self, name, llm_client):
        self.name = name
        self.llm = llm_client

    def _llm_call(self, system_prompt, user_message, temperature=None):
        """Make an LLM call with system + user prompt, return text."""
        return self.llm.chat_with_system(system_prompt, user_message, temperature)

    def _llm_call_json(self, system_prompt, user_message, temperature=None):
        """Make an LLM call expecting JSON response."""
        return self.llm.chat_json(system_prompt, user_message, temperature)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Planning Agent — Graph-of-Thought Query Decomposition (GoTQD)
#  Concept #4: decomposes complex queries into a DAG of sub-queries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PLANNER_PROMPT = """You are the NEXUS Planning Agent. Analyze queries and decide how to decompose them.

Respond with valid JSON only:
{
    "query_type": "factual|comparison|multi-hop|temporal|creative|analytical",
    "complexity": "simple|moderate|complex",
    "needs_decomposition": true/false,
    "sub_queries": [
        {"id": "Q1", "query": "sub-question", "depends_on": [], "strategy": "vector|keyword|web|vector+web|vector+keyword"}
    ],
    "reasoning": "brief explanation"
}

Rules:
1. Simple factual → no decomposition, single sub_query
2. Comparisons → decompose into parts being compared
3. Multi-hop → decompose into steps with dependencies
4. Independent sub-queries have empty depends_on (run in parallel)"""


class PlanningAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("Planner", llm_client)

    def execute(self, query, memory_context=""):
        prompt = f"Query: {query}"
        if memory_context:
            prompt += f"\n\nRelevant context from memory:\n{memory_context}"

        result = self._llm_call_json(PLANNER_PROMPT, prompt, temperature=0.2)

        if "error" in result:
            return {
                "query_type": "factual", "complexity": "simple",
                "needs_decomposition": False,
                "sub_queries": [{"id": "Q1", "query": query, "depends_on": [], "strategy": "vector+web"}],
            }
        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Vector Retriever — semantic search using ChromaDB
#  Part of SPRF (Concept #3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VectorRetriever(BaseAgent):
    """Semantic similarity search over ChromaDB vector store."""

    def __init__(self, llm_client, retrieval_config, embedding_config, vectorstore_path="data/vectorstore"):
        super().__init__("VectorRetriever", llm_client)
        self.config = retrieval_config
        self.vectorstore_path = vectorstore_path

        os.makedirs(vectorstore_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=vectorstore_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="nexus_documents", metadata={"hnsw:space": "cosine"},
        )
        self.embedding_engine = EmbeddingEngine(embedding_config)

    def execute(self, query, top_k=None):
        """Search for semantically similar documents."""
        k = top_k or self.config.max_results
        try:
            query_embedding = self.embedding_engine.embed(query)
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=k,
                include=["documents", "metadatas", "distances"],
            )
            documents = []
            for i in range(len(results["ids"][0])):
                dist = results["distances"][0][i] if results["distances"] else 0
                documents.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": dist,
                    "relevance": 1 - dist,
                    "source_type": "vector",
                })
            return {"status": "success", "documents": documents}
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return {"status": "error", "documents": []}

    def ingest_documents(self, documents):
        """Add documents to ChromaDB with embeddings."""
        if not documents:
            return
        ids = [doc["id"] for doc in documents]
        contents = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        embeddings = self.embedding_engine.embed_batch(contents)
        self.collection.upsert(ids=ids, documents=contents, metadatas=metadatas, embeddings=embeddings)

    def get_document_count(self):
        return self.collection.count()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Keyword Retriever — BM25 lexical search
#  Part of SPRF (Concept #3): catches exact term matches
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class KeywordRetriever(BaseAgent):
    """BM25-based lexical search over the document corpus."""

    def __init__(self, llm_client, retrieval_config, index_path="data/memory/bm25_index.json"):
        super().__init__("KeywordRetriever", llm_client)
        self.config = retrieval_config
        self.index_path = index_path
        self.documents = []
        self.bm25 = None
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path) as f:
                    self.documents = json.load(f)
                if self.documents:
                    tokenized = [doc["content"].lower().split() for doc in self.documents]
                    self.bm25 = BM25Okapi(tokenized)
            except Exception:
                pass

    def _save_index(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "w") as f:
            json.dump(self.documents, f)

    def execute(self, query, top_k=None):
        """Search using BM25 lexical matching."""
        k = top_k or self.config.max_results
        if not self.bm25 or not self.documents:
            return {"status": "empty", "documents": []}

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        scored_docs = sorted(zip(scores, self.documents), key=lambda x: x[0], reverse=True)

        documents = []
        max_score = scored_docs[0][0] if scored_docs else 1.0
        for score, doc in scored_docs[:k]:
            documents.append({
                "id": doc.get("id", "unknown"),
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
                "relevance": score / max_score if max_score > 0 else 0,
                "bm25_score": float(score),
                "source_type": "keyword",
            })
        return {"status": "success", "documents": documents}

    def ingest_documents(self, documents):
        """Add documents to the BM25 index."""
        self.documents.extend(documents)
        tokenized = [doc["content"].lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        self._save_index()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Web Retriever — live web search via Tavily
#  Part of SPRF (Concept #3): handles real-time queries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class WebRetriever(BaseAgent):
    """Web search agent using Tavily API."""

    def __init__(self, llm_client, retrieval_config):
        super().__init__("WebRetriever", llm_client)
        self.config = retrieval_config
        self.tavily_client = None
        if retrieval_config.tavily_api_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=retrieval_config.tavily_api_key)
            except ImportError:
                logger.warning("tavily-python not installed. Web search disabled.")

    def execute(self, query, top_k=None):
        """Search the web for live information."""
        k = top_k or self.config.max_results
        if not self.tavily_client:
            return {"status": "disabled", "documents": []}

        try:
            response = self.tavily_client.search(query=query, max_results=k, include_raw_content=False)
            documents = []
            for result in response.get("results", []):
                documents.append({
                    "id": f"web_{hash(result.get('url', '')) % 10000}",
                    "content": result.get("content", ""),
                    "metadata": {"url": result.get("url", ""), "title": result.get("title", ""),
                                 "source": "web_search"},
                    "relevance": result.get("score", 0.5),
                    "source_type": "web",
                })
            return {"status": "success", "documents": documents}
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {"status": "error", "documents": []}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Reciprocal Rank Fusion — merges results from multiple retrievers
#  Core of SPRF (Concept #3): documents confirmed by multiple
#  sources get higher scores
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RRF_K = 60  # Standard RRF constant from the original paper


def reciprocal_rank_fusion(result_lists, k=RRF_K):
    """
    Merge multiple ranked lists using RRF.
    For each doc: score = sum(1 / (k + rank_in_list)) across all lists.
    Documents appearing in multiple lists naturally get boosted.
    """
    rrf_scores = defaultdict(float)
    doc_map = {}
    source_counts = defaultdict(int)

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, 1):
            doc_id = doc.get("id", str(hash(doc.get("content", ""))))
            rrf_scores[doc_id] += 1.0 / (k + rank)
            source_counts[doc_id] += 1
            # Keep the richest version
            if doc_id not in doc_map or len(doc.get("content", "")) > len(doc_map[doc_id].get("content", "")):
                doc_map[doc_id] = doc

    fused = []
    for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = round(score, 6)
        doc["source_count"] = source_counts[doc_id]
        fused.append(doc)
    return fused


def apply_diversity_penalty(documents, max_per_source=3):
    """Prevent any single retriever from dominating the results."""
    source_counts = defaultdict(int)
    diversified = []
    for doc in documents:
        source_type = doc.get("source_type", "unknown")
        if source_counts[source_type] < max_per_source:
            diversified.append(doc)
        else:
            doc_copy = doc.copy()
            doc_copy["rrf_score"] = doc_copy.get("rrf_score", 0) * 0.5
            diversified.append(doc_copy)
        source_counts[source_type] += 1
    diversified.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
    return diversified


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Adversarial Verification Parliament (Concept #2)
#  Three agents debate document quality before answers are generated
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADVOCATE_PROMPT = """You are the ADVOCATE in the NEXUS Verification Parliament.
Argue FOR the relevance and accuracy of retrieved documents.

Respond in JSON:
{"argument": "why these docs are good", "relevant_docs": ["id1"],
 "supporting_evidence": ["fact1"], "quality_score": 0.85,
 "strengths": ["strength1"]}"""

PROSECUTOR_PROMPT = """You are the PROSECUTOR in the NEXUS Verification Parliament.
CHALLENGE the retrieved documents and find flaws.

Respond in JSON:
{"argument": "issues found", "irrelevant_docs": ["id1"],
 "issues_found": [{"type": "outdated|irrelevant", "doc_id": "id", "detail": "..."}],
 "gaps": ["missing aspect"], "severity_score": 0.3,
 "weaknesses": ["weakness1"]}"""

JUDGE_PROMPT = """You are the JUDGE in the NEXUS Verification Parliament.
Weigh Advocate (FOR) and Prosecutor (AGAINST) arguments.

Verdicts: ACCEPT (docs are good) | PARTIAL (drop some) | REJECT (re-retrieve)

Respond in JSON:
{"decision": "ACCEPT|PARTIAL|REJECT", "reasoning": "...",
 "approved_doc_ids": [], "rejected_doc_ids": [],
 "scores": {"relevance": 0.85, "accuracy": 0.8, "recency": 0.7, "completeness": 0.75},
 "refined_query": "better query if REJECT", "gaps_to_fill": []}"""


class AdvocateAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("Advocate", llm_client)

    def execute(self, query, documents):
        docs_text = "\n\n".join([
            f"[Doc {doc.get('id', i)}] ({doc.get('source_type', 'unknown')})\n{doc.get('content', '')[:500]}"
            for i, doc in enumerate(documents)
        ])
        prompt = f"USER QUERY: {query}\n\nRETRIEVED DOCUMENTS:\n{docs_text}"
        result = self._llm_call_json(ADVOCATE_PROMPT, prompt, temperature=0.3)
        return {"role": "advocate", **result}


class ProsecutorAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("Prosecutor", llm_client)

    def execute(self, query, documents):
        docs_text = "\n\n".join([
            f"[Doc {doc.get('id', i)}] ({doc.get('source_type', 'unknown')})\n{doc.get('content', '')[:500]}"
            for i, doc in enumerate(documents)
        ])
        prompt = f"USER QUERY: {query}\n\nRETRIEVED DOCUMENTS:\n{docs_text}"
        result = self._llm_call_json(PROSECUTOR_PROMPT, prompt, temperature=0.4)
        return {"role": "prosecutor", **result}


class JudgeAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("Judge", llm_client)

    def execute(self, query, advocate_result, prosecutor_result, documents):
        prompt = f"""USER QUERY: {query}

ADVOCATE'S ARGUMENT:
{advocate_result.get('argument', 'No argument')}
Quality Score: {advocate_result.get('quality_score', 'N/A')}
Strengths: {advocate_result.get('strengths', [])}

PROSECUTOR'S ARGUMENT:
{prosecutor_result.get('argument', 'No argument')}
Severity Score: {prosecutor_result.get('severity_score', 'N/A')}
Issues: {prosecutor_result.get('issues_found', [])}
Gaps: {prosecutor_result.get('gaps', [])}

NUMBER OF DOCUMENTS: {len(documents)}
DOCUMENT IDS: {[d.get('id', 'unknown') for d in documents]}

Render your verdict."""

        result = self._llm_call_json(JUDGE_PROMPT, prompt, temperature=0.2)
        return {
            "role": "judge",
            "decision": result.get("decision", "ACCEPT"),
            "judge_reasoning": result.get("reasoning", ""),
            "advocate_summary": advocate_result.get("argument", "")[:200],
            "prosecutor_summary": prosecutor_result.get("argument", "")[:200],
            "approved_doc_ids": result.get("approved_doc_ids", []),
            "rejected_doc_ids": result.get("rejected_doc_ids", []),
            "scores": result.get("scores", {}),
            "refined_query": result.get("refined_query", ""),
            "gaps_to_fill": result.get("gaps_to_fill", []),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Synthesis Agent — generates final answer from verified context
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYNTHESIS_PROMPT = """You are the NEXUS Synthesis Agent. Produce a comprehensive answer from verified sources.

Rules:
1. Use ONLY information from the provided documents
2. Include inline citations like [Source 1], [Source 2]
3. Use clear structure with headings and bullets
4. If docs don't fully answer the query, state what's missing
5. Use markdown formatting"""


class SynthesisAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("Synthesizer", llm_client)

    def execute(self, query, documents, computation_results=None, memory_context=""):
        docs_text = "\n\n".join([
            f"[Source {i+1}] ({doc.get('source_type', 'unknown')}) "
            f"{'[URL: ' + doc.get('metadata', {}).get('url', '') + ']' if doc.get('metadata', {}).get('url') else ''}\n"
            f"{doc.get('content', '')[:800]}"
            for i, doc in enumerate(documents)
        ])
        prompt = f"USER QUERY: {query}\n\nVERIFIED SOURCES:\n{docs_text}"

        if computation_results:
            comp = "\n".join([f"- {cr.get('tool', '')}: {cr.get('result', '')}" for cr in computation_results])
            prompt += f"\n\nCOMPUTATION RESULTS:\n{comp}"
        if memory_context:
            prompt += f"\n\nPRIOR KNOWLEDGE:\n{memory_context}"

        answer = self._llm_call(SYNTHESIS_PROMPT, prompt, temperature=0.3)
        sources = [
            {"title": doc.get("metadata", {}).get("title", f"Source {i+1}"),
             "type": doc.get("source_type", "doc"),
             "relevance": doc.get("relevance", 0.5),
             "url": doc.get("metadata", {}).get("url", "")}
            for i, doc in enumerate(documents)
        ]
        return {"answer": answer, "sources": sources, "num_sources": len(documents)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Confidence Agent — Epistemic Uncertainty Quantification (EUQ)
#  Concept #6: multi-dimensional confidence scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONFIDENCE_PROMPT = """You are the NEXUS Confidence Calibrator. Score the answer's reliability.

Evaluate (0.0 to 1.0 each):
- factual_accuracy, source_agreement, temporal_relevance, coverage

Also identify: flags (issues), unknowns (gaps), suggestions (follow-ups)

Respond in JSON:
{"factual_accuracy": 0.85, "source_agreement": 0.90, "temporal_relevance": 0.65,
 "coverage": 0.78, "overall": 0.80,
 "flags": [], "unknowns": [], "suggestions": []}"""


class ConfidenceAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("Confidence", llm_client)

    def execute(self, query, answer, sources, parliament_scores=None):
        source_summary = "\n".join([
            f"- {s.get('title', 'Unknown')} ({s.get('type', 'doc')}, relevance: {s.get('relevance', 0):.0%})"
            for s in sources
        ])
        prompt = f"QUERY: {query}\n\nANSWER:\n{answer[:1500]}\n\nSOURCES:\n{source_summary}\n\nPARLIAMENT: {parliament_scores or 'N/A'}"

        result = self._llm_call_json(CONFIDENCE_PROMPT, prompt, temperature=0.2)

        # Compute overall if missing
        if "overall" not in result or not isinstance(result.get("overall"), (int, float)):
            dims = ["factual_accuracy", "source_agreement", "temporal_relevance", "coverage"]
            scores = [result.get(d, 0.5) for d in dims]
            result["overall"] = sum(scores) / len(scores)
        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tool Smith — Self-Evolving Tool Synthesis (SETS)
#  Concept #7: dynamically generates Python tools when needed
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOLSMITH_PROMPT = """You are the NEXUS Tool Smith. Analyze if computation is needed.

Respond in JSON:
{"needs_computation": true/false, "computation_type": "math|data_processing|none",
 "can_use_existing": true/false, "existing_tool": "tool_name or null",
 "tool_code": "python function code if new tool needed",
 "tool_name": "function_name"}

Rules: Only generate for actual computation. Pure Python only. Include error handling."""


class ToolSmithAgent(BaseAgent):
    """Generates, tests, and registers Python tools on-the-fly."""

    def __init__(self, llm_client, tools_dir="nexus/tools/generated"):
        super().__init__("ToolSmith", llm_client)
        self.tools_dir = tools_dir
        self.registered_tools = {}
        os.makedirs(tools_dir, exist_ok=True)
        # Built-in computation tools
        self.registered_tools["calculate"] = lambda expr: eval(expr)
        self.registered_tools["percentage"] = lambda part, whole: round((part / whole) * 100, 2) if whole else 0

    def execute(self, query, context=""):
        prompt = f"QUERY: {query}\nCONTEXT: {context[:500] if context else 'None'}\n\nDoes this need computation?"
        result = self._llm_call_json(TOOLSMITH_PROMPT, prompt, temperature=0.2)

        if not result.get("needs_computation", False):
            return {"status": "no_computation_needed", "results": []}

        # Try existing tool
        if result.get("can_use_existing") and result.get("existing_tool") in self.registered_tools:
            return {"status": "used_existing_tool", "tool": result["existing_tool"], "results": []}

        # Generate and register new tool
        tool_code = result.get("tool_code", "")
        tool_name = result.get("tool_name", "custom_tool")

        if tool_code:
            try:
                namespace = {}
                exec(tool_code, namespace)
                if tool_name in namespace:
                    self.registered_tools[tool_name] = namespace[tool_name]
                    # Save for reuse
                    path = os.path.join(self.tools_dir, f"{tool_name}.py")
                    with open(path, "w") as f:
                        f.write(f'"""Auto-generated tool: {tool_name}"""\n\n{tool_code}\n')
                    return {"status": "tool_created", "tool": tool_name, "results": []}
            except Exception as e:
                logger.error(f"Tool synthesis failed: {e}")
                return {"status": "synthesis_failed", "error": str(e), "results": []}

        return {"status": "no_tool_needed", "results": []}
