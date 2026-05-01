import json
import os
import pytest
from pathlib import Path
from src.data_utils.data_loader import DataLoader


@pytest.fixture
def temp_json_intents(tmp_path):
    json_file = tmp_path / "intents.json"
    data = {
        "intents": [
            {
                "tag": "greeting",
                "patterns": ["Hi", "Hello"],
                "responses": ["Hello!", "Hi there!"]
            },
            {
                "tag": "goodbye",
                "patterns": ["Bye", "See you"],
                "responses": ["Goodbye!"]
            }
        ]
    }
    json_file.write_text(json.dumps(data))
    return str(json_file)


@pytest.fixture
def temp_json_list(tmp_path):
    json_file = tmp_path / "data.json"
    data = [
        {"id": 0, "text": "sample 1"},
        {"id": 1, "text": "sample 2"}
    ]
    json_file.write_text(json.dumps(data))
    return str(json_file)


@pytest.fixture
def temp_txt_file(tmp_path):
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("Line one\nLine two\n\nLine three\n")
    return str(txt_file)


class TestDataLoader:
    def test_init_with_valid_path(self, temp_json_intents):
        loader = DataLoader(temp_json_intents)
        assert loader.data_source_path == Path(temp_json_intents)

    def test_init_with_missing_path(self):
        with pytest.raises(ValueError, match="data_source_path must be provided"):
            DataLoader()

    def test_init_with_nonexistent_path(self):
        with pytest.raises(FileNotFoundError):
            DataLoader("/nonexistent/path/data.json")

    def test_load_json_intents(self, temp_json_intents):
        loader = DataLoader(temp_json_intents)
        result = loader.load_content()

        assert len(result) == 2
        assert result[0]['tag'] == 'greeting'
        assert 'Hi' in result[0]['patterns']
        assert 'Hello!' in result[0]['responses']
        assert result[1]['tag'] == 'goodbye'

    def test_load_json_list(self, temp_json_list):
        loader = DataLoader(temp_json_list)
        result = loader.load_content()

        assert len(result) == 2
        assert result[0]['text'] == 'sample 1'

    def test_load_txt_file(self, temp_txt_file):
        loader = DataLoader(temp_txt_file)
        result = loader.load_content()

        assert len(result) == 3
        assert result[0]['content'] == 'Line one'
        assert result[1]['content'] == 'Line two'
        assert result[2]['content'] == 'Line three'

    def test_unsupported_format(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n")

        with pytest.raises(ValueError, match="Unsupported file format"):
            DataLoader(str(csv_file)).load_content()

    def test_intents_content_combination(self, temp_json_intents):
        loader = DataLoader(temp_json_intents)
        result = loader.load_content()

        assert 'Hi' in result[0]['content']
        assert 'Hello!' in result[0]['content']
