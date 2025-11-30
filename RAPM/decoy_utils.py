import random

def replace_entity_with_wrong_one(text: str, gold_entities: list[str], candidate_pool: list[str], rng) -> str:
    """
    Pick one gold entity and swap it with a different entity from a global pool.
    """
    if not gold_entities or not candidate_pool:
        return text

    # Filter gold entities that are actually in the text
    present_gold_entities = [e for e in gold_entities if e in text]
    if not present_gold_entities:
        return text

    target_ent = rng.choice(present_gold_entities)
    # choose a different entity as wrong replacement
    wrong_choices = [e for e in candidate_pool if e != target_ent]
    if not wrong_choices:
        return text

    wrong_ent = rng.choice(wrong_choices)
    return text.replace(target_ent, wrong_ent, 1)  # replace only first occurrence


def add_distractor_passage(contexts: list, distractor: str, position: str = "front") -> list:
    """
    Insert a distractor passage before or after the gold contexts.
    contexts: list of dicts (retrieved_info) or strings.
    """
    if not contexts:
        # If contexts is empty, we create a list with just the distractor
        # We assume the caller expects the same type as contexts would have
        # But since we don't know the type if it's empty, we might default to dict structure used in RAG_prompt_cons
        return [{"db_label": distractor, "confidence level": "<Low Confidence>", "faiss_score": 0.0, "mmseqs_score": 0.0, "id": -1}]

    is_dict = isinstance(contexts[0], dict)
    
    distractor_item = distractor
    if is_dict:
        distractor_item = {
            "db_label": distractor,
            "confidence level": "<Low Confidence>",
            "faiss_score": 0.0,
            "mmseqs_score": 0.0,
            "id": -1
        }

    if position == "front":
        return [distractor_item] + contexts
    return contexts + [distractor_item]


def sample_random_distractor(sample_metadata, distractor_db, rng):
    """
    sample_metadata: metadata of the current query
    distractor_db: list of {'text': str, 'metadata': ...}
    """
    # Extract gold entities
    gold_entities = set()
    if isinstance(sample_metadata, dict):
        gold_entities = set(sample_metadata.get("bio_entity", []))
    elif isinstance(sample_metadata, str):
        if len(sample_metadata) < 100:
             gold_entities = {sample_metadata}

    candidates = []
    # Try to find distractors that don't share entities
    # We'll sample a subset to avoid iterating everything if DB is huge
    db_indices = list(range(len(distractor_db)))
    rng.shuffle(db_indices)
    
    for idx in db_indices[:100]: # Check 100 random candidates
        row = distractor_db[idx]
        row_ents = set()
        if isinstance(row['metadata'], dict):
            row_ents = set(row['metadata'].get("bio_entity", []))
        elif isinstance(row['metadata'], str):
             if len(row['metadata']) < 100:
                row_ents = {row['metadata']}
        
        if not row_ents.isdisjoint(gold_entities):
            continue
        
        candidates.append(row['text'])
        if len(candidates) >= 10: # Found enough
            break

    if not candidates:
        # fallback: just use any random text
        candidates = [distractor_db[i]['text'] for i in db_indices[:5]]

    if not candidates:
        return "Irrelevant protein function."

    return rng.choice(candidates)
