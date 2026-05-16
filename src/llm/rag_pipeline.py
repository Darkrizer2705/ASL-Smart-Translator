# src/llm/rag_pipeline.py
import numpy as np
import anthropic
import os
import json

# ── Knowledge base ─────────────────────────────────
# This is your RAG document store — ASL phrase contexts
ASL_KNOWLEDGE_BASE = [
    {
        "id": 1,
        "category": "medical",
        "keywords": ["hospital", "help", "pain", "doctor", "medicine"],
        "context": "Medical emergency ASL phrases. Common signs: HELP, HOSPITAL, PAIN, DOCTOR, MEDICINE, AMBULANCE.",
        "examples": [
            {"raw": "ME HELP HOSPITAL", "refined": "I need help at the hospital."},
            {"raw": "PAIN ME STOMACH", "refined": "I have stomach pain."},
            {"raw": "DOCTOR WHERE", "refined": "Where is the doctor?"},
        ]
    },
    {
        "id": 2,
        "category": "greetings",
        "keywords": ["hello", "goodbye", "thankyou", "please", "sorry", "name"],
        "context": "Basic greeting and courtesy ASL phrases. Common signs: HELLO, GOODBYE, THANK-YOU, PLEASE, SORRY, NAME.",
        "examples": [
            {"raw": "HELLO NAME ME", "refined": "Hello, my name is..."},
            {"raw": "THANKYOU PLEASE", "refined": "Thank you, please."},
            {"raw": "SORRY UNDERSTAND NO", "refined": "I'm sorry, I don't understand."},
        ]
    },
    {
        "id": 3,
        "category": "daily_needs",
        "keywords": ["water", "food", "bathroom", "home", "school", "work"],
        "context": "Daily needs and locations in ASL. Common signs: WATER, FOOD, BATHROOM, HOME, SCHOOL, WORK.",
        "examples": [
            {"raw": "WATER ME PLEASE", "refined": "May I have some water please?"},
            {"raw": "BATHROOM WHERE", "refined": "Where is the bathroom?"},
            {"raw": "FOOD ME WANT", "refined": "I want food."},
        ]
    },
    {
        "id": 4,
        "category": "questions",
        "keywords": ["what", "where", "when", "why", "how", "who"],
        "context": "Question words in ASL. Common signs: WHAT, WHERE, WHEN, WHY, HOW, WHO.",
        "examples": [
            {"raw": "YOU WHERE GO", "refined": "Where are you going?"},
            {"raw": "WHAT ME DO", "refined": "What should I do?"},
            {"raw": "WHEN HOME YOU", "refined": "When are you going home?"},
        ]
    },
    {
        "id": 5,
        "category": "family_social",
        "keywords": ["family", "friend", "you", "me", "he_she", "understand"],
        "context": "Family and social interaction in ASL. Common signs: FAMILY, FRIEND, YOU, ME, HE, SHE.",
        "examples": [
            {"raw": "FAMILY HOME", "refined": "My family is at home."},
            {"raw": "FRIEND YOU ME", "refined": "You are my friend."},
            {"raw": "HE_SHE UNDERSTAND NO", "refined": "He/She doesn't understand."},
        ]
    },
    {
        "id": 6,
        "category": "emotions",
        "keywords": ["happy", "sad", "angry", "stop", "more", "again", "wait"],
        "context": "Emotions and states in ASL. Common signs: HAPPY, SAD, ANGRY, STOP, MORE, AGAIN, WAIT.",
        "examples": [
            {"raw": "ME HAPPY", "refined": "I am happy."},
            {"raw": "STOP PLEASE", "refined": "Please stop."},
            {"raw": "MORE AGAIN", "refined": "Do it again, more."},
        ]
    },
]

# ── Simple TF-IDF style retrieval ──────────────────
def retrieve_context(raw_words: list, top_k: int = 2) -> list:
    """
    Retrieves most relevant knowledge base entries
    based on keyword overlap with signed words.
    This is the RETRIEVAL part of RAG.
    """
    raw_lower  = [w.lower() for w in raw_words]
    scores     = []

    for doc in ASL_KNOWLEDGE_BASE:
        # Count keyword matches
        keyword_matches = sum(
            1 for kw in doc["keywords"]
            if kw in raw_lower
        )
        # Count example matches
        example_matches = sum(
            1 for ex in doc["examples"]
            if any(w in ex["raw"].lower() for w in raw_lower)
        )
        score = keyword_matches * 2 + example_matches
        scores.append((score, doc))

    # Sort by score, return top_k
    scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scores[:top_k] if score > 0]


def build_rag_prompt(raw_words: list, retrieved_docs: list) -> str:
    """
    Builds the augmented prompt with retrieved context.
    This is the AUGMENTATION part of RAG.
    """
    raw_text = " ".join(raw_words).upper()

    # Build context from retrieved documents
    context_str = ""
    for doc in retrieved_docs:
        context_str += f"\nCategory: {doc['category']}\n"
        context_str += f"Context: {doc['context']}\n"
        context_str += "Similar examples:\n"
        for ex in doc["examples"]:
            context_str += f"  Raw: {ex['raw']} → Refined: {ex['refined']}\n"

    if not context_str:
        context_str = "No specific context found. Use general ASL interpretation."

    prompt = f"""You are an expert ASL (American Sign Language) interpreter.

RETRIEVED CONTEXT FROM KNOWLEDGE BASE:
{context_str}

TASK:
Convert the following raw ASL sign sequence into natural, grammatically correct English.
Use the retrieved context and examples above to inform your interpretation.

Raw ASL signs: {raw_text}

Rules:
- Output ONLY the refined English sentence
- No explanations, no quotes, no extra text  
- Make it natural and contextually appropriate
- Use the retrieved examples as reference

Natural English:"""

    return prompt


def rag_refine(raw_words: list) -> dict:
    """
    Full RAG pipeline:
    1. RETRIEVE relevant context from knowledge base
    2. AUGMENT the prompt with retrieved context
    3. GENERATE refined sentence using LLM

    Returns dict with full pipeline details for transparency.
    """
    if not raw_words:
        return {"raw": "", "retrieved": [], "refined": ""}

    raw_text = " ".join(raw_words).upper()

    # ── Step 1: RETRIEVE ───────────────────────────
    retrieved_docs = retrieve_context(raw_words)
    print(f"\nRetrieved {len(retrieved_docs)} relevant documents")
    for doc in retrieved_docs:
        print(f"   -> {doc['category']}: {doc['context'][:60]}...")

    # ── Step 2: AUGMENT ────────────────────────────
    prompt = build_rag_prompt(raw_words, retrieved_docs)
    print(f"\nAugmented prompt built ({len(prompt)} chars)")

    # ── Step 3: GENERATE ───────────────────────────
    client   = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")
    )
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    refined = response.content[0].text.strip()
    print(f"\nGenerated: {refined}")

    return {
        "raw":       raw_text,
        "retrieved": [doc["category"] for doc in retrieved_docs],
        "refined":   refined,
        "prompt":    prompt
    }


# ── Test standalone ────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ["me", "help", "hospital"],
        ["water", "please", "me"],
        ["family", "home", "food"],
        ["what", "you", "name"],
        ["stop", "again", "understand", "no"],
        ["happy", "me", "friend", "you"],
    ]

    print("RAG Pipeline Test")
    print("=" * 60)

    for words in test_cases:
        result = rag_refine(words)
        print(f"\nRaw:       {result['raw']}")
        print(f"Retrieved: {result['retrieved']}")
        print(f"Refined:   {result['refined']}")
        print("-" * 60)
