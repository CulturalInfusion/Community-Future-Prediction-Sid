"""Basic tests that don't require any external dependencies."""


def test_pytest_discovery():
    """Test that pytest can discover and run tests."""
    assert True


def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    assert 1 + 1 == 2
    assert 2 * 3 == 6
    assert 10 - 5 == 5
    assert 15 / 3 == 5


def test_string_operations():
    """Test basic string operations."""
    text = "Hello, World!"
    assert len(text) == 13
    assert "Hello" in text
    assert text.upper() == "HELLO, WORLD!"
    assert text.lower() == "hello, world!"


def test_list_operations():
    """Test basic list operations."""
    numbers = [1, 2, 3, 4, 5]
    assert len(numbers) == 5
    assert sum(numbers) == 15
    assert max(numbers) == 5
    assert min(numbers) == 1


def test_dict_operations():
    """Test basic dictionary operations."""
    data = {"name": "test", "value": 42}
    assert len(data) == 2
    assert "name" in data
    assert data["value"] == 42
    assert data.get("name") == "test" 
