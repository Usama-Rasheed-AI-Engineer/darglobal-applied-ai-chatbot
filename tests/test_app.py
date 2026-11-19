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


def test_aida_property_types_do_not_return_wasalt():
    response = client.post("/api/chat", json={"message": "What types of properties are available at AIDA?"})
    answer = response.json()["answer"]
    assert "Wasalt" not in answer
    assert "villas" in answer.lower()
    assert "apartments" in answer.lower()
