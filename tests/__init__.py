# placeholder for tests
import json
import pytest
from Deployment import app  # your Flask app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c

def make_dummy_input():
    # 20 dummy years of representation, e.g. [1.0, 2.0, …]
    return [float(i % 10) for i in range(20)]

def test_homepage(client):
    """GET / should return the UI page (status 200)."""
    resp = client.get('/')
    assert resp.status_code == 200
    assert b'<title>' in resp.data

def test_predict_valid(client):
    """POST /predict with valid payload should return 2025–2035 predictions."""
    payload = {
        "community": "african",
        "input": make_dummy_input()
    }
    resp = client.post('/predict',
                       data=json.dumps(payload),
                       content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    # should have 'predictions' key mapping 11 years
    assert 'predictions' in data
    years = list(data['predictions'].keys())
    assert len(years) == 11
    # each value should be a float
    for val in data['predictions'].values():
        assert isinstance(val, float)

def test_predict_too_short(client):
    """Bad request if input list != 20."""
    payload = {"community": "asian", "input": [1,2,3]}
    resp = client.post('/predict',
                       data=json.dumps(payload),
                       content_type='application/json')
    assert resp.status_code == 400
    data = resp.get_json()
    assert 'error' in data

def test_predict_unsupported_community(client):
    """Bad request if unknown community."""
    payload = {"community": "martian", "input": make_dummy_input()}
    resp = client.post('/predict',
                       data=json.dumps(payload),
                       content_type='application/json')
    assert resp.status_code == 400
    data = resp.get_json()
    assert "not supported" in data['error'].lower()
