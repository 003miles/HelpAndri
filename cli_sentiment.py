import os
import uuid
import shutil
import argparse
import pandas as pd
from llm_handler import analyse_sentiements

UPLOAD_DIR = "temp_uploads"
RESULT_DIR = "temp_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def main():
    defaults = {
        "columns": "title,quotes",
        "prompt": "How does the text portray migrants, immigrants, asylum seekers, or ethnic minorities?",
        "choices": "positive,negative,neutral",
        "sample": 1000,
        "workers": 4
    }

    parser = argparse.ArgumentParser(description="Run sentiment classification on a CSV using an LLM.")
    parser.add_argument("--file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--columns", type=str, default=defaults["columns"], required=True, help="Comma-separated list of columns to analyze.")
    parser.add_argument("--prompt", type=str, default=defaults["prompt"], required=True, help="Classification prompt.")
    parser.add_argument("--choices", type=str, default=defaults["choices"], required=True, help="Comma-separated sentiment choices (e.g. positive,negative,neutral).")
    parser.add_argument("--sample", type=int, default=defaults["sample"], help="Sample size for dataset (default 1000)")
    parser.add_argument("--workers", type=int, default=defaults["workers"], help="Max workers for cpu threading (default 4)")

    args = parser.parse_args()

    # Prepare inputs
    input_path = args.file
    file_id = str(uuid.uuid4())
    filename = os.path.basename(input_path)
    upload_path = os.path.join(UPLOAD_DIR, f"{file_id}_{filename}")
    shutil.copy(input_path, upload_path)
    workers = args.workers

    columns = [col.strip() for col in args.columns.split(",")]
    choices = [c.strip() for c in args.choices.split(",")]

    try:
        df = pd.read_csv(upload_path)
    except Exception as e:
        print(f"❌ CSV parsing error: {e}")
        return

    if not all(col in df.columns for col in columns):
        print("❌ One or more selected columns do not exist in the CSV.")
        print("Available columns:", list(df.columns))
        return

    # Optional sampling
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42)

    df["__combined_text"] = df[columns].astype(str).agg(" ".join, axis=1)
    sentiments = analyse_sentiements(df["__combined_text"].tolist(), args.prompt, choices, max_workers=workers)
    df["Sentiment"] = sentiments

    output_path = os.path.join(RESULT_DIR, f"annotated_{filename}")
    df.to_csv(output_path, index=False)
    print(f"✅ Sentiment analysis complete! Output saved to: {output_path}")


if __name__ == "__main__":
    main()