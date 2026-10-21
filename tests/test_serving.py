"""Production serving endpoint tests — includes streaming validation."""
import json, asyncio, pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Mock tokenizer for tests
class FakeTokenizer:
    eos_token_id = 2
    def encode(self, text): return [1] + [ord(c) % 100 for c in text[:20]] + [2]
    def decode(self, ids): return "".join(chr(i + 32) if 20 <= i + 32 < 127 else " " for i in ids)

# Mock model
class FakeModel:
    cfg = MagicMock()
    def __init__(self): self.cfg.vocab_size = 32000
    def eval(self): return self
    def __call__(self, *a, **kw):
        import torch
        B, T = a[0].shape
        return torch.randn(B, T, 32000)
    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        import torch
        new = torch.randint(3, 1000, (input_ids.shape[0], max_new_tokens))
        return torch.cat([input_ids, new], dim=1)
    def num_parameters(self): return 100_000_000

@pytest.fixture
def app():
    from open_mythos.serving.server import create_app
    return create_app(model=FakeModel(), tokenizer=FakeTokenizer(), device="cpu")

@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app, headers={"X-API-Key": "dev"})

def test_liveness(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"

def test_readiness(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"

def test_models_list(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1

def test_completion_non_stream(client):
    r = client.post("/v1/completions", json={
        "model": "openmythos", "prompt": "Hello world", "max_tokens": 5, "stream": False
    })
    assert r.status_code == 200
    body = r.json()
    assert "choices" in body
    assert len(body["choices"]) == 1
    assert "usage" in body
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["completion_tokens"] > 0

def test_completion_streaming():
    """Verify SSE stream emits tokens and terminates with [DONE]."""
    from open_mythos.serving.server import create_app
    from fastapi.testclient import TestClient
    app = create_app(model=FakeModel(), tokenizer=FakeTokenizer(), device="cpu")
    client = TestClient(app, headers={"X-API-Key": "dev"})

    with client.stream("POST", "/v1/completions", json={
        "model": "openmythos", "prompt": "test", "max_tokens": 5, "stream": True
    }) as resp:
        assert resp.status_code == 200
        content_type = resp.headers.get("content-type", "")
        assert "text/event-stream" in content_type
        raw = resp.read().decode()
        assert "data:" in raw
        assert "[DONE]" in raw

def test_chat_completion(client):
    r = client.post("/v1/chat/completions", json={
        "model": "openmythos",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 10,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["role"] == "assistant"

def test_auth_required():
    from open_mythos.serving.server import create_app
    from fastapi.testclient import TestClient
    import os
    os.environ["OPENMYTHOS_API_KEYS"] = "valid-key-only"
    # Reload module to pick up new env
    import importlib, open_mythos.serving.server as sm
    importlib.reload(sm)
    app = sm.create_app(model=FakeModel(), tokenizer=FakeTokenizer(), device="cpu")
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 5})
    assert r.status_code == 401
    del os.environ["OPENMYTHOS_API_KEYS"]

def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
