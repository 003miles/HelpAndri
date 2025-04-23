from typing import List
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm


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

    normalised_choices = {choice.lower(): choice for choice in choices}

    for line in reversed(output_clean.splitlines()):
        if line.startswith("answer:"):
            answer_text = line.replace("answer:", "").strip()
            if answer_text in normalised_choices:
                return normalised_choices[answer_text]

            for key in normalised_choices:
                if key in answer_text:
                    return normalised_choices[key]

    # Fallback: match with first known label found anywhere in output
    for key in normalised_choices:
        if key in output_clean:
            return normalised_choices[key]

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


def test():
    test_strings = [
        "Migrants have significantly contributed to the UK economy.",
        "This article highlights the positive cultural impact of migration.",
        "The new policy supports integrating migrants into the workforce.",
        "Local communities have welcomed migrants with open arms.",
        "Migration enriches society and brings diversity.",
        "Migrants are putting pressure on public services.",
        "The report blames rising crime rates on recent migration.",
        "This article portrays migrants as a threat to national identity.",
        "Illegal immigration is described as out of control.",
        "Migrants are accused of taking jobs from locals.",
        "The government released new migration statistics today.",
        "The article outlines changes in migration laws without opinion.",
        "A timeline of major UK migration events is provided.",
        "The piece discusses both benefits and challenges of migration.",
        "Migration numbers have steadily increased since 2010.",
        "While migrants help the economy, the article also raises concerns about integration.",
        "The headline praises diversity, but the tone feels cautious.",
        "Quotes from experts are supportive, but reader comments are critical.",
        "The article presents contrasting views on immigration policy.",
        "Migration is described as a complex and divisive issue."
    ]

    choices = ["positive", "negative", "neutral"]

    prompt = """
    The sentence is taken from a newspaper article on migration, 
    in what light does it portray migrants, immigrants, asylum seekers, or ethnic minorities? 
    """

    output_sentiments = analyse_sentiments(test_strings, prompt, choices, model="deepseek-r1:1.5b", max_workers=4, debug=True)

    result = zip(test_strings, output_sentiments)

    for r in result:
        print(tuple(r))


if __name__ == "__main__":
    test()
