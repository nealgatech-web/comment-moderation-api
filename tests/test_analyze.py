from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_analyze_basic():
    r = client.post("/analyze", json={"texts":["I hate you","hello"]})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data and len(data["results"]) == 2
    assert all(0.0 <= x["toxicity"] <= 1.0 for x in data["results"])

def test_train_and_analyze():
    payload = {
        "samples":[
            {"text":"you suck","label":"toxic"},
            {"text":"good job","label":"neutral"},
            {"text":"thanks","label":"neutral"},
            {"text":"go to hell","label":"toxic"}
        ]
    }
    r = client.post("/train", json=payload)
    assert r.status_code == 200
    r2 = client.post("/analyze", json={"texts":["you suck","hi there"]})
    assert r2.status_code == 200
