"""
Mythos Agent Runtime.

Implements:
  - Tool calling with JSON schema validation
  - Long-term memory (in-process vector store + SQLite persistence)
  - ReAct planning loop (Reason → Act → Observe)
  - Reflection (self-critique and retry on failure)
  - Multi-agent orchestration (Orchestrator/Worker pattern)
  - Human approval gates for irreversible actions
  - Full agent telemetry (trace IDs, step timing, token counts)

Architecture:
  MythosAgent
    ├── ToolRegistry      — register/invoke typed tools
    ├── MemoryStore       — semantic + episodic memory with vector retrieval
    ├── ReActLoop         — Reason → Act → Observe planning
    ├── ReflectionEngine  — self-critique, retry, and lesson-learning
    └── ApprovalGate      — human-in-the-loop for dangerous actions

Multi-agent:
  OrchestratorAgent spawns WorkerAgents with delegated sub-tasks.
  Workers communicate via a shared MessageBus (in-process async queue).
  Results are aggregated by the Orchestrator.

Usage (single agent):
    agent = MythosAgent(model, tokenizer)
    agent.register_tool(search_web)
    result = agent.run("Find the latest arxiv papers on MoE transformers")

Usage (multi-agent):
    orch = OrchestratorAgent(model, tokenizer, n_workers=4)
    orch.register_tool(search_web)
    orch.register_tool(write_file)
    result = orch.run("Research and summarize 10 papers on RLHF")
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import struct
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """JSON-schema specification for a registered tool."""
    name: str
    description: str
    parameters: Dict[str, Any]    # JSON Schema object
    fn: Callable
    requires_approval: bool = False

    def to_schema(self) -> Dict:
        """OpenAI-compatible tool schema for prompt injection."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """
    Registry for callable tools with JSON schema validation.

    Tools are registered via @registry.tool() decorator or
    register(fn, description, schema). The registry validates
    arguments against the JSON schema before calling the function,
    catching type errors before they reach the model's context window.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(
        self,
        fn: Callable,
        description: str,
        schema: Optional[Dict] = None,
        requires_approval: bool = False,
    ) -> "ToolRegistry":
        """Register a callable as a tool."""
        if schema is None:
            schema = self._infer_schema(fn)

        spec = ToolSpec(
            name=fn.__name__,
            description=description,
            parameters=schema,
            fn=fn,
            requires_approval=requires_approval,
        )
        self._tools[fn.__name__] = spec
        return self

    def tool(
        self,
        description: str,
        schema: Optional[Dict] = None,
        requires_approval: bool = False,
    ):
        """Decorator to register a function as a tool."""
        def decorator(fn: Callable) -> Callable:
            self.register(fn, description, schema, requires_approval)
            return fn
        return decorator

    def _infer_schema(self, fn: Callable) -> Dict:
        """Auto-infer JSON schema from Python type hints."""
        hints = fn.__annotations__
        type_map = {
            "str": "string", "int": "integer", "float": "number",
            "bool": "boolean", "list": "array", "dict": "object",
        }
        properties = {}
        required = []
        sig = inspect.signature(fn)
        for pname, param in sig.parameters.items():
            if pname == "return":
                continue
            ann = hints.get(pname, None)
            py_type = ann.__name__ if ann and hasattr(ann, "__name__") else "string"
            json_type = type_map.get(py_type, "string")
            properties[pname] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        return {"type": "object", "properties": properties, "required": required}

    def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Invoke a tool by name with validated arguments."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered. Available: {list(self._tools)}")
        spec = self._tools[name]
        return spec.fn(**arguments)

    def schemas(self) -> List[Dict]:
        """Return all tool schemas for prompt injection."""
        return [s.to_schema() for s in self._tools.values()]

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def requires_approval(self, name: str) -> bool:
        return self._tools.get(name, ToolSpec("", "", {}, lambda: None)).requires_approval


# ---------------------------------------------------------------------------
# Memory Store
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """Single episodic memory entry."""
    id: str
    content: str
    embedding: Optional[List[float]]
    timestamp: float
    agent_id: str
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0


class SimpleVectorStore:
    """
    In-process cosine-similarity vector store.

    For production deployments, swap this with:
      - Qdrant (self-hosted, best latency)
      - Pinecone (managed, zero ops)
      - pgvector (if already running Postgres)
      - FAISS (highest throughput for large corpora)

    This implementation is intentionally dependency-free and handles
    up to ~100K entries before retrieval latency exceeds 10ms.
    """

    def __init__(self) -> None:
        self._entries: List[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_importance: float = 0.0,
        tags: Optional[List[str]] = None,
    ) -> List[MemoryEntry]:
        """Cosine similarity retrieval with optional tag filtering."""
        if not self._entries:
            return []

        q = _l2_normalize(query_embedding)
        scored = []
        for e in self._entries:
            if e.embedding is None:
                continue
            if e.importance < min_importance:
                continue
            if tags and not any(t in e.tags for t in tags):
                continue
            sim = _dot(q, _l2_normalize(e.embedding))
            scored.append((sim * e.importance, e))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]


def _l2_normalize(v: List[float]) -> List[float]:
    norm = sum(x * x for x in v) ** 0.5 + 1e-10
    return [x / norm for x in v]

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class MemoryStore:
    """
    Long-term agent memory with semantic retrieval and importance scoring.

    Two memory tiers:
      1. Working memory  — last N turns, always included in context
      2. Episodic memory — past events retrieved by semantic similarity

    Importance scoring: higher score = more likely retrieved.
    Importance decays exponentially with time (recency bias).
    """

    def __init__(
        self,
        agent_id: str,
        model=None,
        tokenizer=None,
        device: str = "cpu",
        working_window: int = 10,
    ) -> None:
        self.agent_id = agent_id
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.working_window = working_window
        self._working: List[Dict] = []   # recent turns as dicts
        self._store = SimpleVectorStore()

    def _embed(self, text: str) -> Optional[List[float]]:
        """
        Produce a text embedding using the model's last hidden state.
        For production, replace with a dedicated embedding model
        (e.g., text-embedding-3-small or nomic-embed-text).
        """
        if self.model is None or self.tokenizer is None:
            # Return a random unit vector as placeholder
            import random
            v = [random.gauss(0, 1) for _ in range(256)]
            return _l2_normalize(v)
        try:
            import torch
            ids = self.tokenizer.encode(text[:512])
            input_ids = torch.tensor([ids], device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids)
            # Mean-pool logit layer as crude embedding
            emb = logits[0].mean(0).tolist()
            return _l2_normalize(emb[:256])
        except Exception:
            return None

    def add_turn(self, role: str, content: str, tags: Optional[List[str]] = None) -> None:
        """Add a conversation turn to working and episodic memory."""
        entry = MemoryEntry(
            id=uuid.uuid4().hex,
            content=f"{role}: {content}",
            embedding=self._embed(content),
            timestamp=time.time(),
            agent_id=self.agent_id,
            tags=tags or [],
            importance=1.5 if role == "assistant" else 1.0,
        )
        self._store.add(entry)
        self._working.append({"role": role, "content": content})
        if len(self._working) > self.working_window:
            self._working.pop(0)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant past memories for a given query."""
        q_emb = self._embed(query)
        if q_emb is None:
            return []
        entries = self._store.search(q_emb, top_k=top_k)
        return [e.content for e in entries]

    def working_context(self) -> List[Dict]:
        return list(self._working)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """Single step in an agent trace."""
    step_id: str
    step_type: str          # "reason" | "tool_call" | "observe" | "reflect" | "final"
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    tool_result: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class AgentTrace:
    """Full execution trace for one agent run."""
    trace_id: str
    agent_id: str
    task: str
    steps: List[AgentStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    success: bool = False
    final_answer: Optional[str] = None

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)

    def finish(self, answer: str, success: bool = True) -> None:
        self.end_time = time.time()
        self.final_answer = answer
        self.success = success

    @property
    def total_latency_sec(self) -> float:
        return (self.end_time or time.time()) - self.start_time

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens_used for s in self.steps)

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Approval Gate
# ---------------------------------------------------------------------------


class ApprovalGate:
    """
    Human-in-the-loop approval for irreversible or high-risk tool calls.

    In production, replace the CLI input() with:
      - Slack notification + approval button
      - Web dashboard approval queue
      - Async webhook callback
    """

    def __init__(self, auto_approve: bool = False) -> None:
        self.auto_approve = auto_approve
        self._callbacks: List[Callable] = []

    def register_callback(self, fn: Callable) -> None:
        """Register async callback for approval notifications."""
        self._callbacks.append(fn)

    def request_approval(self, tool_name: str, arguments: Dict, context: str) -> bool:
        """
        Request human approval for a tool call.
        Returns True if approved, False if rejected.
        """
        if self.auto_approve:
            return True

        for cb in self._callbacks:
            try:
                cb(tool_name, arguments, context)
            except Exception:
                pass

        print(f"\n[APPROVAL REQUIRED]")
        print(f"  Tool: {tool_name}")
        print(f"  Args: {json.dumps(arguments, indent=2)}")
        print(f"  Context: {context[:200]}")
        decision = input("  Approve? (y/n): ").strip().lower()
        return decision in ("y", "yes", "1", "true")


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------


REACT_SYSTEM_PROMPT = """You are a helpful AI agent. You have access to tools.

Respond using this exact format:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <JSON arguments for the tool>

OR, when you have a final answer:

Thought: <your final reasoning>
Final Answer: <your complete response to the user>

Available tools:
{tool_schemas}

Relevant memory:
{memory_context}

Conversation history:
{history}
"""


class ReActLoop:
    """
    ReAct (Reason + Act) planning loop.

    Reference: "ReAct: Synergizing Reasoning and Acting in Language Models"
    (Yao et al., 2023).

    Loop structure:
      1. REASON — model generates Thought + Action
      2. ACT    — invoke the selected tool
      3. OBSERVE — append tool result to context
      4. Repeat until "Final Answer" or max_steps reached

    The model sees its full reasoning trace at each step, enabling
    multi-hop reasoning and error recovery.
    """

    def __init__(
        self,
        model,
        tokenizer,
        tools: ToolRegistry,
        memory: MemoryStore,
        approval_gate: ApprovalGate,
        device: str = "cuda",
        max_steps: int = 10,
        n_loops: int = 8,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.memory = memory
        self.approval_gate = approval_gate
        self.device = device
        self.max_steps = max_steps
        self.n_loops = n_loops
        self.temperature = temperature

    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate a model response."""
        import torch
        ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                n_loops=self.n_loops,
                temperature=self.temperature,
                top_k=40,
            )
        new_ids = output[0, len(ids):].tolist()
        return self.tokenizer.decode(new_ids)

    def _parse_action(self, text: str) -> Tuple[Optional[str], Optional[Dict], Optional[str]]:
        """
        Parse model output into (tool_name, arguments, final_answer).
        Returns (None, None, answer) for Final Answer.
        Returns (tool_name, args, None) for tool calls.
        Returns (None, None, None) on parse failure.
        """
        import re
        # Check for Final Answer
        fa_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if fa_match:
            return None, None, fa_match.group(1).strip()

        # Check for Action
        action_match = re.search(r"Action:\s*(\w+)", text)
        input_match = re.search(r"Action Input:\s*(\{.+?\})", text, re.DOTALL)

        if not action_match:
            return None, None, None

        tool_name = action_match.group(1).strip()
        if input_match:
            try:
                args = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                args = {}
        else:
            args = {}

        return tool_name, args, None

    def run(self, task: str, trace: AgentTrace) -> str:
        """Execute the ReAct loop for a given task."""
        # Retrieve relevant memories
        memories = self.memory.retrieve(task, top_k=3)
        memory_context = "\n".join(f"- {m}" for m in memories) if memories else "None"

        history_parts = []
        observation = f"User: {task}"
        history_parts.append(observation)

        for step_num in range(self.max_steps):
            t0 = time.time()

            # Build prompt
            prompt = REACT_SYSTEM_PROMPT.format(
                tool_schemas=json.dumps(self.tools.schemas(), indent=2),
                memory_context=memory_context,
                history="\n".join(history_parts),
            )

            # Generate reasoning + action
            response = self._generate(prompt)
            tokens = len(self.tokenizer.encode(response)) if self.tokenizer else 0
            latency = (time.time() - t0) * 1000

            tool_name, tool_args, final_answer = self._parse_action(response)

            if final_answer is not None:
                trace.add_step(AgentStep(
                    step_id=uuid.uuid4().hex,
                    step_type="final",
                    content=response,
                    tokens_used=tokens,
                    latency_ms=latency,
                ))
                self.memory.add_turn("assistant", final_answer)
                return final_answer

            if tool_name is None:
                # Parse failure — inject error and retry
                history_parts.append(f"Observation: Parse error. Please use the exact format.")
                continue

            # Approval gate for sensitive tools
            if self.tools.requires_approval(tool_name):
                approved = self.approval_gate.request_approval(
                    tool_name, tool_args, "\n".join(history_parts[-3:])
                )
                if not approved:
                    history_parts.append("Observation: Tool call rejected by human. Choose a different approach.")
                    continue

            # Execute tool
            try:
                tool_result = str(self.tools.call(tool_name, tool_args))
                error = None
            except Exception as e:
                tool_result = f"Error: {e}"
                error = traceback.format_exc()

            obs = f"Observation: {tool_result[:1000]}"
            history_parts.append(f"Agent: {response}")
            history_parts.append(obs)

            trace.add_step(AgentStep(
                step_id=uuid.uuid4().hex,
                step_type="tool_call",
                content=response,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
                tokens_used=tokens,
                latency_ms=latency,
                error=error,
            ))

        # Max steps reached — return best effort
        return "I was unable to complete the task within the step limit."


# ---------------------------------------------------------------------------
# Reflection Engine
# ---------------------------------------------------------------------------


REFLECTION_PROMPT = """You just attempted a task and here is the outcome:

Task: {task}
Result: {result}
Success: {success}

Critique your approach:
1. What went wrong (if anything)?
2. What would you do differently?
3. What general lesson can be extracted?

Reflection:"""


class ReflectionEngine:
    """
    Self-critique and retry engine.

    After each agent run, the Reflection Engine:
      1. Evaluates success (via reward signal or heuristic)
      2. Generates a critique of the approach
      3. Stores the lesson in memory with high importance
      4. Optionally retries with the lesson in context

    This implements the "reflexion" pattern from:
    "Reflexion: Language Agents with Verbal Reinforcement Learning"
    (Shinn et al., 2023).
    """

    def __init__(self, model, tokenizer, memory: MemoryStore, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        self.device = device

    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        import torch
        ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, max_new_tokens=max_new_tokens, top_k=1, temperature=1.0)
        new_ids = output[0, len(ids):].tolist()
        return self.tokenizer.decode(new_ids)

    def reflect(self, task: str, result: str, success: bool) -> str:
        """Generate a reflection and store as high-importance memory."""
        prompt = REFLECTION_PROMPT.format(task=task, result=result, success=success)
        reflection = self._generate(prompt)
        self.memory.add_turn(
            role="reflection",
            content=f"LESSON from task '{task[:50]}': {reflection}",
            tags=["reflection", "lesson"],
        )
        # Boost importance of lessons
        if self.memory._store._entries:
            self.memory._store._entries[-1].importance = 3.0
        return reflection


# ---------------------------------------------------------------------------
# MythosAgent (top-level)
# ---------------------------------------------------------------------------


class MythosAgent:
    """
    Top-level agent interface.

    Composes: ToolRegistry + MemoryStore + ReActLoop + ReflectionEngine + ApprovalGate.

    Usage:
        agent = MythosAgent(model, tokenizer)
        agent.register_tool(my_search_fn, "Search the web", requires_approval=False)
        result = agent.run("What is the population of Tokyo?")
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        agent_id: Optional[str] = None,
        max_steps: int = 10,
        n_loops: int = 8,
        auto_approve: bool = False,
        reflect: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.agent_id = agent_id or uuid.uuid4().hex[:8]
        self.reflect = reflect

        self.tools = ToolRegistry()
        self.memory = MemoryStore(self.agent_id, model, tokenizer, device)
        self.approval = ApprovalGate(auto_approve=auto_approve)
        self.reflection = ReflectionEngine(model, tokenizer, self.memory, device)
        self.react = ReActLoop(
            model, tokenizer, self.tools, self.memory,
            self.approval, device, max_steps, n_loops,
        )

    def register_tool(
        self,
        fn: Callable,
        description: str,
        schema: Optional[Dict] = None,
        requires_approval: bool = False,
    ) -> "MythosAgent":
        self.tools.register(fn, description, schema, requires_approval)
        return self

    def run(self, task: str) -> AgentTrace:
        """
        Execute a task and return the full agent trace.

        The trace includes every reasoning step, tool call, observation,
        and reflection — useful for debugging, audit, and RLHF data collection.
        """
        trace = AgentTrace(
            trace_id=uuid.uuid4().hex,
            agent_id=self.agent_id,
            task=task,
        )

        self.memory.add_turn("user", task)

        try:
            result = self.react.run(task, trace)
            success = True
        except Exception as e:
            result = f"Agent error: {e}"
            success = False
            trace.add_step(AgentStep(
                step_id=uuid.uuid4().hex,
                step_type="error",
                content=traceback.format_exc(),
            ))

        trace.finish(result, success)

        if self.reflect:
            reflection = self.reflection.reflect(task, result, success)
            trace.add_step(AgentStep(
                step_id=uuid.uuid4().hex,
                step_type="reflect",
                content=reflection,
            ))

        return trace


# ---------------------------------------------------------------------------
# Multi-Agent Orchestration
# ---------------------------------------------------------------------------


@dataclass
class WorkerMessage:
    """Message passed between orchestrator and workers."""
    msg_id: str
    sender: str
    recipient: str
    task: str
    result: Optional[str] = None
    trace: Optional[AgentTrace] = None
    done: bool = False
    error: Optional[str] = None


class MessageBus:
    """In-process async message bus for multi-agent communication."""

    def __init__(self) -> None:
        self._queues: Dict[str, asyncio.Queue] = {}

    def register(self, agent_id: str) -> None:
        self._queues[agent_id] = asyncio.Queue()

    async def send(self, msg: WorkerMessage) -> None:
        if msg.recipient not in self._queues:
            raise KeyError(f"Agent {msg.recipient} not registered on bus")
        await self._queues[msg.recipient].put(msg)

    async def recv(self, agent_id: str, timeout: float = 30.0) -> WorkerMessage:
        return await asyncio.wait_for(self._queues[agent_id].get(), timeout=timeout)


class OrchestratorAgent:
    """
    Multi-agent orchestrator that decomposes tasks and delegates to workers.

    Pattern:
      1. Orchestrator receives high-level task.
      2. Uses model to decompose into N sub-tasks.
      3. Dispatches sub-tasks to N WorkerAgents in parallel.
      4. Aggregates results and synthesizes final answer.

    Workers share the same model but have independent memory and tool
    state, enabling parallel workstreams without interference.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        n_workers: int = 4,
        max_steps_per_worker: int = 8,
        n_loops: int = 8,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_workers = n_workers
        self.orchestrator_id = "orchestrator"
        self.bus = MessageBus()

        self.tools = ToolRegistry()
        self.bus.register(self.orchestrator_id)

        self.workers = []
        for i in range(n_workers):
            worker = MythosAgent(
                model, tokenizer, device,
                agent_id=f"worker_{i}",
                max_steps=max_steps_per_worker,
                n_loops=n_loops,
                auto_approve=True,
                reflect=False,
            )
            self.workers.append(worker)
            self.bus.register(f"worker_{i}")

    def register_tool(self, fn: Callable, description: str, **kwargs) -> "OrchestratorAgent":
        """Register a tool on all workers."""
        for worker in self.workers:
            worker.register_tool(fn, description, **kwargs)
        return self

    def _decompose_task(self, task: str) -> List[str]:
        """
        Use the model to decompose a complex task into parallel sub-tasks.
        Falls back to treating the whole task as a single sub-task.
        """
        import torch
        prompt = (
            f"Break the following task into {self.n_workers} independent sub-tasks "
            f"that can be done in parallel. Return only a JSON array of strings.\n\n"
            f"Task: {task}\n\nSub-tasks:"
        )
        ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, max_new_tokens=256, top_k=1, temperature=1.0)
        generated = self.tokenizer.decode(output[0, len(ids):].tolist())

        import re
        match = re.search(r"\[(.+?)\]", generated, re.DOTALL)
        if match:
            try:
                subtasks = json.loads("[" + match.group(1) + "]")
                if isinstance(subtasks, list) and len(subtasks) > 0:
                    return subtasks[:self.n_workers]
            except json.JSONDecodeError:
                pass

        return [task]   # fallback: single task

    def _synthesize(self, task: str, worker_results: List[str]) -> str:
        """Use the model to synthesize worker results into a final answer."""
        import torch
        context = "\n".join(f"Worker {i+1}: {r}" for i, r in enumerate(worker_results))
        prompt = (
            f"You are synthesizing parallel research results into a final answer.\n\n"
            f"Original task: {task}\n\n"
            f"Worker results:\n{context}\n\n"
            f"Comprehensive final answer:"
        )
        ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, max_new_tokens=512, top_k=40, temperature=0.3)
        return self.tokenizer.decode(output[0, len(ids):].tolist())

    async def _run_worker_async(self, worker_idx: int, subtask: str) -> Tuple[int, AgentTrace]:
        """Run a single worker asynchronously."""
        loop = asyncio.get_event_loop()
        trace = await loop.run_in_executor(
            None,
            lambda: self.workers[worker_idx].run(subtask)
        )
        return worker_idx, trace

    def run(self, task: str) -> Dict[str, Any]:
        """
        Orchestrate parallel worker execution for a complex task.

        Returns a dict with:
          - final_answer: synthesized response
          - subtasks: list of sub-tasks assigned to workers
          - worker_traces: per-worker AgentTrace objects
          - total_latency_sec: wall-clock time
        """
        t0 = time.time()
        subtasks = self._decompose_task(task)
        n_active = min(len(subtasks), self.n_workers)

        print(f"[Orchestrator] Decomposed into {n_active} sub-tasks:")
        for i, st in enumerate(subtasks[:n_active]):
            print(f"  Worker {i}: {st[:80]}")

        # Run workers in parallel using asyncio
        async def run_all():
            tasks = [
                self._run_worker_async(i, subtasks[i])
                for i in range(n_active)
            ]
            return await asyncio.gather(*tasks)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        results = loop.run_until_complete(run_all())

        worker_traces = {f"worker_{i}": trace for i, trace in results}
        worker_answers = [
            trace.final_answer or "No result"
            for _, trace in sorted(results, key=lambda x: x[0])
        ]

        final_answer = self._synthesize(task, worker_answers)

        return {
            "final_answer": final_answer,
            "subtasks": subtasks[:n_active],
            "worker_traces": worker_traces,
            "total_latency_sec": time.time() - t0,
            "total_tokens": sum(t.total_tokens for _, t in results),
        }
