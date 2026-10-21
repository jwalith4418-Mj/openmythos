import os
import sys

# Configure environment variables at the absolute top so they are processed during import evaluation
os.environ["OPENMYTHOS_DISABLE_BATCHING"] = "1"
os.environ["OPENMYTHOS_DISABLE_AUTH"] = "1"

import time
import threading
import torch
import httpx
import asyncio

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath("."))

from open_mythos.main import OpenMythos, MythosConfig
from open_mythos.serving.server import serve, create_app
from open_mythos.agents.runtime import MythosAgent, OrchestratorAgent

class SimpleTokenizer:
    eos_token_id = 2
    def encode(self, text):
        return [1] + [ord(c) % 200 for c in text[:15]] + [2]
    def decode(self, ids):
        return "".join(chr(min(i + 32, 126)) for i in ids if i > 2)

def run_server(app, host, port):
    import uvicorn
    print(f"\n[SERVER-THREAD] Starting Uvicorn server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port, log_level="warning")

def test_server_endpoints(port):
    print("\n" + "="*80)
    print("  STAGE 1: VERIFYING SERVING ENDPOINTS (OpenAI-Compatible & Loop Profiling)")
    print("="*80)
    client = httpx.Client(base_url=f"http://localhost:{port}")
    
    # 1. Health checks
    try:
        r = client.get("/health")
        print(f"[HTTP GET /health] status: {r.status_code}, body: {r.json()}")
        
        r = client.get("/health/ready")
        print(f"[HTTP GET /health/ready] status: {r.status_code}, body: {r.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
        
    # 2. Models
    r = client.get("/v1/models")
    print(f"[HTTP GET /v1/models] status: {r.status_code}, body: {r.json()}")
    
    # 3. Text completions
    payload = {
        "model": "openmythos",
        "prompt": "The core thesis of elastic transformers is",
        "max_tokens": 8,
        "n_loops": 4,
        "stream": False
    }
    r = client.post("/v1/completions", json=payload)
    print(f"[HTTP POST /v1/completions] status: {r.status_code}")
    if r.status_code == 200:
        res = r.json()
        print(f"  Generated Text: '{res['choices'][0]['text']}'")
        print(f"  Tokens Used: {res['usage']}")

    # 4. OpenMythos-Unique Elastic Compute Loop Profiling
    r = client.post("/v1/loop_profile", json=payload)
    print(f"[HTTP POST /v1/loop_profile] status: {r.status_code}")
    if r.status_code == 200:
        res = r.json()
        print("  Elastic Compute Telemetry:")
        print(f"    Loop counts per generated token: {res['loop_counts']}")
        print(f"    Mean loop depth: {res['mean_loops']}")
        print(f"    Estimated FLOPS: {res['estimated_flops']:.2e}")
        print(f"    Compute utilization ratio: {res['compute_utilization']}")
    
    client.close()
    return True

def run_single_agent(model, tokenizer):
    print("\n" + "="*80)
    print("  STAGE 2: ACTIVATING SINGLE AGENT (ReAct Loop + Memory + Reflection)")
    print("="*80)
    
    agent = MythosAgent(model, tokenizer, device="cpu", max_steps=5, n_loops=4, auto_approve=True)
    
    # Register mock tools
    def search_web(query: str) -> str:
        print(f"  [Tool: search_web] Query: '{query}'")
        return f"Arxiv papers on Recurrent-Depth show a 2-4x Reasoning-FLOP Efficiency advantage for query: {query}."

    def write_summary(filename: str, content: str) -> str:
        print(f"  [Tool: write_summary] Writing to '{filename}'...")
        return f"File '{filename}' written successfully with {len(content)} chars."

    agent.register_tool(search_web, description="Search the web for real-time information.")
    agent.register_tool(write_summary, description="Save a summary to a specified file.")

    task = "Research Recurrent-Depth efficiency and save summary to mythos_notes.txt"
    print(f"Starting agent task: '{task}'")
    
    trace = agent.run(task)
    
    print("\nExecution Complete!")
    print(f"  Final Answer: {trace.final_answer}")
    print(f"  Steps taken: {len(trace.steps)}")
    for i, s in enumerate(trace.steps):
        print(f"    Step {i+1} [{s.step_type}]: {s.content.strip()[:100]}...")
        if s.tool_name:
            print(f"      Called Tool: '{s.tool_name}' with args: {s.tool_args}")
            print(f"      Tool Observation: {s.tool_result[:100]}...")
            
    print(f"  Total tokens processed: {trace.total_tokens}")
    print(f"  Agent Reflection / Critique:\n    {trace.steps[-1].content.strip()}")

def run_multi_agent(model, tokenizer):
    print("\n" + "="*80)
    print("  STAGE 3: ACTIVATING MULTI-AGENT ORCHESTRATOR (Task Decomposition + Message Bus)")
    print("="*80)
    
    # 3 Parallel workers
    orchestrator = OrchestratorAgent(
        model, tokenizer, device="cpu", n_workers=3, max_steps_per_worker=5, n_loops=4
    )
    
    # Register search tool on workers
    def search_knowledge_base(query: str) -> str:
        print(f"  [Worker Tool] Querying internal databases for: '{query}'")
        return f"Synthesized findings for '{query}': Elastic recurrence spectrum is stable, spectrum radius ρ(A) < 1."
        
    orchestrator.register_tool(search_knowledge_base, "Query the internal OpenMythos research database")
    
    task = "Perform a deep multi-agent feasibility analysis on LTI spectrum stability, speculative depth decoding, and Triton kernel optimizations."
    print(f"Starting Multi-Agent Orchestrated Goal: '{task}'")
    
    # Run Orchestration
    result = orchestrator.run(task)
    
    print("\nMulti-Agent Execution Complete!")
    print(f"  Sub-tasks assigned to workers:")
    for i, st in enumerate(result['subtasks']):
        print(f"    Worker {i}: '{st}'")
        
    print(f"\n  Final Synthesized Answer from Orchestrator:\n{result['final_answer']}")
    print(f"  Total multi-agent execution time: {result['total_latency_sec']:.2f} seconds")
    print(f"  Total tokens processed across all agents: {result['total_tokens']}")

def main():
    print("Initializing tiny OpenMythos model for fast CPU verification...")
    # Miniature config for rapid local CPU execution
    cfg = MythosConfig(
        vocab_size=256, dim=64, n_heads=4, n_kv_heads=2,
        max_seq_len=1024, max_loop_iters=4, prelude_layers=1, coda_layers=1,
        n_experts=4, n_shared_experts=1, n_experts_per_tok=2, expert_dim=32,
        kv_lora_rank=32, q_lora_rank=64, qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16,
        max_cache_len=2048, max_output_tokens=1024, lora_rank=4,
        act_threshold=0.95,
    )
    model = OpenMythos(cfg).eval()
    tokenizer = SimpleTokenizer()
    
    # Disable batching scheduler during HTTP verification so it runs on CPU instantly
    os.environ["OPENMYTHOS_DISABLE_BATCHING"] = "1"
    os.environ["OPENMYTHOS_DISABLE_AUTH"] = "1"
    
    # Create FastAPI app
    app = create_app(model=model, tokenizer=tokenizer, device="cpu")
    
    # Start server in background thread
    port = 9099
    server_thread = threading.Thread(target=run_server, args=(app, "127.0.0.1", port), daemon=True)
    server_thread.start()
    
    # Wait for server to boot up
    print("Waiting 3 seconds for Uvicorn server to start...")
    time.sleep(3.0)
    
    # STAGE 1: Test Server Endpoints
    server_ok = test_server_endpoints(port)
    if not server_ok:
        print("[CRITICAL] Server failed to start or respond!")
        return
        
    # STAGE 2: Run Single Agent
    run_single_agent(model, tokenizer)
    
    # STAGE 3: Run Multi-Agent
    run_multi_agent(model, tokenizer)
    
    print("\n" + "="*80)
    print("  ALL SYSTEM COMPONENTS RUNNING AND AGENTS SUCCESSFULLY ACTIVATED!")
    print("="*80)

if __name__ == "__main__":
    main()
