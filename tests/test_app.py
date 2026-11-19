from fastapi.testclient import TestClient

from app.main import app
from app.retrieval import chunk_text

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
    assert "darglobal.co.uk/landing-page/aida" in answer


def test_conversation_memory_is_returned_and_capped():
    first = client.post("/api/chat", json={"message": "Tell me about AIDA"})
    session_id = first.json()["session_id"]
    second = client.post("/api/chat", json={"message": "What properties are available there?", "session_id": session_id})
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id


def test_explicit_project_switch_does_not_return_aida():
    first = client.post("/api/chat", json={"message": "Tell me about AIDA"})
    session_id = first.json()["session_id"]
    second = client.post(
        "/api/chat",
        json={"message": "What types of properties are available at Tierra Viva?", "session_id": session_id},
    )
    answer = second.json()["answer"]
    assert "Tierra Viva" in answer
    assert "AIDA" not in answer


def test_unknown_project_does_not_return_another_project():
    response = client.post("/api/chat", json={"message": "What properties are available at Unknown Project?"})
    answer = response.json()["answer"].lower()
    assert "could not find" in answer


def test_source_chunking_preserves_provenance():
    chunks = chunk_text(
        "AIDA is a gated community. It offers villas and apartments.",
        "https://darglobal.co.uk/landing-page/aida",
        "DarGlobal",
        "AIDA",
        "2026-11-19",
    )
    assert chunks
    assert chunks[0]["url"].endswith("/landing-page/aida")
    assert chunks[0]["source"] == "DarGlobal"
