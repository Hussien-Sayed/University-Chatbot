"""
List available HuggingFace embedding models via API (no extra dependencies).
"""
import requests

url = "https://huggingface.co/api/models"

params = {
    "filter": "feature-extraction",
    "sort": "downloads",
    "direction": "-1",
    "limit": 20
}

response = requests.get(url, params=params)
models = response.json()

print("\nTop 20 available embedding models (sorted by downloads):")
print("-" * 80)
for i, model in enumerate(models, 1):
    print(f"{i:2}. {model['modelId']}")
    print(f"    Downloads: {model.get('downloads', 'N/A'):,}")
    print(f"    Tags: {', '.join(model.get('tags', [])[:3])}")
    print()
