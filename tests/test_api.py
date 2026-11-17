from fastapi.testclient import TestClient

from nexus.api import app

client = TestClient(app)


def test_health_and_agents():
    assert client.get("/health").json()["status"] == "ok"
    agents = client.get("/api/v1/agents").json()
    assert len(agents) == 5


def test_simulation_flow_and_query():
    r = client.post("/api/v1/simulation/start", json={"seed": 42, "max_steps": 30})
    assert r.status_code == 200
    ep = r.json()["episode_id"]
    r2 = client.post(f"/api/v1/simulation/{ep}/step", json={"action": None})
    assert r2.status_code == 200
    assert client.get(f"/api/v1/episodes/{ep}/trajectory").status_code == 200
    assert client.get(f"/api/v1/simulation/{ep}/state").status_code == 200
    q = client.post("/api/v1/query", json={"question": "Why did Buyer A reject?", "episode_id": ep, "agent_id": "buyer_a"})
    assert q.status_code == 200
    body = q.json()
    assert body["evidence"] and 0 <= body["confidence"] <= 1


def test_query_answers_differ_by_question():
    r = client.post("/api/v1/simulation/start", json={"seed": 42, "max_steps": 100})
    ep = r.json()["episode_id"]
    for _ in range(12):
        client.post(f"/api/v1/simulation/{ep}/step", json={"action": None})

    def ask(question, agent_id=None):
        body = {"question": question, "episode_id": ep}
        if agent_id:
            body["agent_id"] = agent_id
        resp = client.post("/api/v1/query", json=body)
        assert resp.status_code == 200
        return resp.json()

    reservation = ask("What is the reservation price for Seller A?", "seller_a")
    assert "90" in reservation["answer"]

    deals = ask("Who got the best deal?")
    assert "deal" in deals["answer"].lower()

    manager = ask("What did the market manager do?")
    assert "market_manager" in manager["answer"]

    summary = ask("How many deals were closed?")
    assert summary["answer"] != deals["answer"]
    assert manager["answer"] != reservation["answer"]


def test_live_negotiation_is_not_a_fixed_same_role_cycle():
    r = client.post("/api/v1/simulation/start", json={"seed": 42, "max_steps": 100})
    assert r.status_code == 200
    state = r.json()["state"]
    roles = {"buyer_a": "buyer", "buyer_b": "buyer", "seller_a": "seller",
             "seller_b": "seller", "market_manager": "manager"}
    seller_a_offer_prices = []
    for _ in range(48):
        prev_offer = state.get("active_offer")
        step = client.post(f"/api/v1/simulation/{state['episode_id']}/step", json={"action": None})
        assert step.status_code == 200
        body = step.json()
        state = body["state"]
        action = body.get("action") or [None]
        if prev_offer and action[0] == 2:
            assert roles[body["agent"]] != roles[prev_offer["offerer"]], (
                f"{body['agent']} accepted an offer from same-role {prev_offer['offerer']}")
        if body.get("agent") == "seller_a" and action[0] in (0, 1) and state.get("active_offer"):
            seller_a_offer_prices.append(state["active_offer"]["price"])
        if body.get("done"):
            break
    assert len(set(seller_a_offer_prices)) >= 3, seller_a_offer_prices


def test_not_found():
    assert client.get("/api/v1/episodes/nope/trajectory").status_code == 404


def test_websocket():
    ep = client.post("/api/v1/simulation/start", json={"seed": 1}).json()["episode_id"]
    with client.websocket_connect(f"/api/v1/simulation/{ep}/stream") as ws:
        ws.send_text("ping")
        assert ws.receive_json()["pong"] is True
