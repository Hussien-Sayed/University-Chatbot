import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union


def generate_testset(
    data_source_path: str,
    test_set_ratio: float = 0.2,
    test_data_path: Optional[Union[str, Path]] = None,
    refined_data_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, str]]:
    if not os.path.exists(data_source_path):
        raise FileNotFoundError(f"Data source not found: {data_source_path}")

    with open(data_source_path, 'r') as f:
        data = json.load(f)

    intents = data.get('intents', [])
    if not intents:
        return []

    if test_set_ratio < 0 or test_set_ratio > 1:
        raise ValueError("test_set_ratio must be between 0 and 1")

    test_data = []
    refined_data = deepcopy(data)
    selected_questions_by_intent = {}

    for intent_index, intent in enumerate(intents):
        patterns = intent.get('patterns', [])
        responses = intent.get('responses', [])
        if not patterns or not responses:
            continue

        sample_size = round(len(patterns) * test_set_ratio)
        if test_set_ratio > 0:
            sample_size = max(1, sample_size)

        selected_questions = random.sample(patterns, min(sample_size, len(patterns)))
        selected_questions_by_intent[intent_index] = set(selected_questions)

        for question in selected_questions:
            ground_truth = random.choice(responses)
            test_data.append({
                "question": question,
                "ground_truth": ground_truth
            })

    if refined_data_path:
        for intent_index, intent in enumerate(refined_data.get('intents', [])):
            selected_questions = selected_questions_by_intent.get(intent_index, set())
            intent['patterns'] = [
                pattern for pattern in intent.get('patterns', [])
                if pattern not in selected_questions
            ]
        save_json(refined_data_path, refined_data)

    if test_data_path:
        save_testset(test_data_path, test_data)

    return test_data


def save_json(data_path: Union[str, Path], data: Union[Dict, List]) -> None:
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_testset(test_data_path: Union[str, Path]) -> List[Dict[str, str]]:
    with open(test_data_path, 'r') as f:
        return json.load(f)


def save_testset(test_data_path: Union[str, Path], test_data: List[Dict[str, str]]) -> None:
    save_json(test_data_path, test_data)
