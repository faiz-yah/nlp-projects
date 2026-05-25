# app/guardrails.py
import re

FARMING_KEYWORDS = {
    'crop', 'crops', 'soil', 'fertilizer', 'fertiliser', 'pesticide',
    'irrigation', 'harvest', 'harvesting', 'plant', 'plants', 'seed',
    'seeds', 'farm', 'farming', 'farmer', 'agriculture', 'agricultural',
    'pest', 'disease', 'weed', 'yield', 'maize', 'rice', 'wheat',
    'cassava', 'tomato', 'vegetable', 'fruit', 'compost', 'manure',
    'nitrogen', 'phosphorus', 'potassium', 'sowing', 'planting',
    'acre', 'hectare', 'drought', 'rainfall', 'climate', 'grow', 'growing',
}

def is_farming_related(question: str) -> bool:
    tokens = set(re.sub(r"[^\w\s]", "", question.lower()).split())
    return len(tokens & FARMING_KEYWORDS) >= 1

def is_valid_response(text: str, min_real_words: int = 3) -> bool:
    words = text.split()
    real_words = [w for w in words if re.match(r'^[a-zA-Z]{3,}$', w)]
    return len(real_words) >= min_real_words

OFF_TOPIC_MSG = (
    "🌱 I'm AgriBot! I specialise in farming and agriculture. "
    "Please ask me about crops, soil, pests, irrigation, harvesting, or any agricultural topics!"
)
FALLBACK_MSG = "I don't have enough information about that in my knowledge base."