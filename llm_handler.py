from typing import List
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
import re
from typing import List

def format_duration(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds > 0 or not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    return " ".join(parts)

def extract_choice(output: str, choices: List[str]) -> str:
    output_clean = output.strip().lower()
    normalized_choices = {choice.lower(): choice for choice in choices}

    # Helper: match valid label from captured group
    def normalize_and_match(label):
        label = label.lower().strip(' "\'“”‘’.,;')
        return normalized_choices.get(label, None)

    # 1. Match answer line with optional quotes around the word
    answer_match = re.search(
        r'answer\s*[:\-]?\s*[\"“”\'‘’]?(?P<word>\w+)[\"“”\'‘’]?',
        output_clean,
        re.IGNORECASE
    )
    if answer_match:
        answer = answer_match.group("word").lower()
        if answer in normalized_choices:
            return normalized_choices[answer]


    # 2. Regex: classification is '<label>' or it can be considered '<label>'
    classification_match = re.search(
        r"(classification\s+is|can\s+be\s+considered|it\s+is)\s+[\"\']?(?P<word>\w+)[\"\']?",
        output_clean,
        re.IGNORECASE
    )
    if classification_match:
        answer = normalize_and_match(classification_match.group("word"))
        if answer:
            return answer


    # 3. Look for standalone line with just the answer (possibly in quotes)
    for line in reversed(output_clean.splitlines()):
        line_stripped = line.strip().strip('"\''"“”‘’")
        if line_stripped in normalized_choices:
            return normalized_choices[line_stripped]

    # 4. Fallback: look for the first match of a valid choice in the full output
    for key in normalized_choices:
        if re.search(rf'\b{re.escape(key)}\b', output_clean):
            return normalized_choices[key]

    return "unknown"

def analyse_sentiments(texts: List[str], user_prompt: str, output_choices: List[str], model: str, max_workers=4, delay=0.2, debug=False) -> List[str]:
    print("Starting analysis...", flush=True)
    def process_text(text):
        start_indiv = time.time()
        # tqdm.write(f"extracting sentiment '{text[:5]}...'")
        prompt = f"""
        You are a machine for newspaper text sentiment classification.
        You will be given a text, and will need to analyse it in order to respond based on the given question.
        You are not asked to have an opinion on the topic, you must infer the sentiment given by the author of the text.
        Give a brief reasoning, to explain your decision.
        You will then respond with your chosen classification, using only ONE of the given choice words.

        Text: {text}
        Question: {user_prompt}
        Classification choices: {output_choices}
        
        YOU MUST RESPOND FOLLOWING THE FORMAT BELOW
        Reasoning:
        Answer:
        """.strip()

        output = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        raw = output["message"]["content"]
        sentiment = extract_choice(raw, output_choices)
        # tqdm.write(f"finished '{text[:5]}...' in {round(time.time() - start_indiv, 2)}s.")
        return sentiment if not debug else raw

    results: List = [None] * len(texts)
    start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        total = len(texts)
        futures = {}
        for i, text in enumerate(texts):
            futures[executor.submit(process_text, text)] = i
            time.sleep(delay)

        for future in tqdm(as_completed(futures), total=total, desc="Analysing sentiments"):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error on prompt {idx}: {e}")
                results[idx] = "error"

    print(f"Finished in {format_duration(time.time() - start)}")
    return results
