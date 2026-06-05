import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class DataLoader:
    """Class for loading content from data sources (JSON, TXT, etc.)"""

    def __init__(self, data_source_path: Optional[str] = None, document_structure_mode: Optional[str] = None):
        self.data_source_path = data_source_path or os.getenv("DATA_SOURCE_PATH")
        if not self.data_source_path:
            raise ValueError("data_source_path must be provided or set in .env as DATA_SOURCE_PATH")

        self.data_source_path = Path(self.data_source_path)
        if not self.data_source_path.exists():
            raise FileNotFoundError(f"Data source not found: {self.data_source_path}")

        self.document_structure_mode = document_structure_mode or os.getenv("DOCUMENT_STRUCTURE_MODE", "structural")

    def load_content(self) -> List[Dict[str, Any]]:
        file_extension = self.data_source_path.suffix.lower()

        if file_extension == '.json':
            return self._load_json()
        elif file_extension == '.txt':
            return self._load_txt()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def _load_json(self) -> List[Dict[str, Any]]:
        if self.document_structure_mode == "non_structural":
            return self._load_json_as_single_document()

        with open(self.data_source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'intents' in data:
            if self.document_structure_mode == "structural-formatted":
                return self._parse_intents_formatted(data['intents'])
            return self._parse_intents(data['intents'])

        if isinstance(data, list):
            return data

        raise ValueError("JSON format not recognized")

    def _load_txt(self) -> List[Dict[str, Any]]:
        documents = []
        with open(self.data_source_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    documents.append({
                        'id': idx,
                        'content': line,
                        'source': str(self.data_source_path)
                    })
        return documents

    def _parse_intents(self, intents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        documents = []
        for idx, intent in enumerate(intents):
            tag = intent.get('tag', f'intent_{idx}')
            patterns = intent.get('patterns', [])
            responses = intent.get('responses', [])

            content = ' '.join(patterns + responses)

            documents.append({
                'id': idx,
                'tag': tag,
                'content': content,
                'patterns': patterns,
                'responses': responses,
                'source': str(self.data_source_path)
            })

        return documents

    def _parse_intents_formatted(self, intents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        documents = []
        for idx, intent in enumerate(intents):
            tag = intent.get('tag', f'intent_{idx}')
            patterns = intent.get('patterns', [])
            responses = intent.get('responses', [])

            formatted_patterns = ' '.join([f'"{p}"' for p in patterns])
            formatted_responses = ' '.join([f'"{r}"' for r in responses])

            content = f"User questions: {formatted_patterns}. Assistant responses: {formatted_responses}."

            documents.append({
                'id': idx,
                'tag': tag,
                'content': content,
                'patterns': patterns,
                'responses': responses,
                'source': str(self.data_source_path)
            })

        return documents

    def _load_json_as_single_document(self) -> List[Dict[str, Any]]:
        with open(self.data_source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        content = json.dumps(data, ensure_ascii=False)

        return [{
            'id': 0,
            'content': content,
            'tag': None,
            'source': str(self.data_source_path),
            'document_structure_mode': 'non_structural'
        }]
