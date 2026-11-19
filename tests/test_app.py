from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_sources():
    response = client.get("/api/sources")
    assert response.status_code == 200
    assert response.json()["sources"]


def test_chat_returns_grounded_answer():
    response = client.post("/api/chat", json={"message": "Which countries does DarGlobal operate in?"})
    assert response.status_code == 200
    assert "DarGlobal" in response.json()["answer"]


def test_chat_rejects_empty_message():
    response = client.post("/api/chat", json={"message": " "})
    assert response.status_code == 400
