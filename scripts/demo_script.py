"""Final demo: backend flow without a browser (dashboard walkthrough is manual).

Run backend + frontend first (see RESULTS.md), then:
  .venv\\Scripts\\python.exe scripts/demo_script.py
"""
from fastapi.testclient import TestClient

from nexus.api import app


def main() -> int:
    c = TestClient(app)
    ep = c.post("/api/v1/simulation/start", json={"seed": 42, "max_steps": 30}).json()["episode_id"]
    print("episode", ep)
    for _ in range(5):
        print(c.post(f"/api/v1/simulation/{ep}/step", json={"action": None}).json()["state"]["step"])
    q = c.post("/api/v1/query", json={"question": "Why did Buyer A reject?", "episode_id": ep, "agent_id": "buyer_a"})
    print(q.json()["answer"])
    print("trajectory steps", len(c.get(f"/api/v1/episodes/{ep}/trajectory").json()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
