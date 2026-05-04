"""
API integration tests using FastAPI TestClient.

These tests verify the health endpoint and the auth middleware.
RAG pipeline tests mock the LLM and vector DB layers.
"""
import pytest
from fastapi.testclient import TestClient
from main import app
from config.settings import get_settings


@pytest.fixture
def client():
    """Provides a synchronous TestClient for the FastAPI app."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    """Returns valid Authorization headers for protected endpoints."""
    settings = get_settings()
    return {"Authorization": f"Bearer {settings.api_key}"}


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
class TestAuthentication:
    def test_query_without_auth_returns_401(self, client):
        response = client.post("/api/v1/query", json={"query": "test"})
        assert response.status_code == 401

    def test_query_with_bad_token_returns_403(self, client):
        headers = {"Authorization": "Bearer wrong-token"}
        response = client.post(
            "/api/v1/query", json={"query": "test"}, headers=headers
        )
        assert response.status_code == 403

    def test_query_with_malformed_header_returns_401(self, client):
        headers = {"Authorization": "Basic some-token"}
        response = client.post(
            "/api/v1/query", json={"query": "test"}, headers=headers
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------
class TestInputValidation:
    def test_empty_query_returns_422(self, client, auth_headers):
        response = client.post(
            "/api/v1/query", json={"query": ""}, headers=auth_headers
        )
        assert response.status_code == 422

    def test_query_too_long_returns_422(self, client, auth_headers):
        long_query = "a" * 2001
        response = client.post(
            "/api/v1/query", json={"query": long_query}, headers=auth_headers
        )
        assert response.status_code == 422

    def test_top_k_out_of_range_returns_422(self, client, auth_headers):
        response = client.post(
            "/api/v1/query",
            json={"query": "valid question", "top_k": 100},
            headers=auth_headers,
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Ingest endpoint — file format validation
# ---------------------------------------------------------------------------
class TestIngestValidation:
    def test_ingest_unsupported_format_returns_400(self, client, auth_headers):
        files = {"file": ("test.csv", b"col1,col2", "text/csv")}
        response = client.post(
            "/api/v1/ingest", files=files, headers=auth_headers
        )
        assert response.status_code == 400
        assert "Unsupported" in response.json()["detail"]
