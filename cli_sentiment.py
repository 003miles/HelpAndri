import os
import uuid
import shutil
import argparse
import pandas as pd
from llm_handler import analyse_sentiments

UPLOAD_DIR = "temp_uploads"
RESULT_DIR = "temp_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def main():
    with open("prompt.txt", "r+") as f:
        file_prompt = f.read()

    defaults = {
        "model": "mistral",
        "columns": "title,quotes",
        "prompt": file_prompt,
        "choices": "positive,negative,neutral,unrelated",
        "sample": 1000,
        "workers": 4
    }

    parser = argparse.ArgumentParser(description="Run sentiment classification on a CSV using an LLM.")
    parser.add_argument("--file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--model", type=str, default=defaults["model"], help="Model to use for sentiment classification.")
    parser.add_argument("--columns", type=str, default=defaults["columns"], help="Comma-separated list of columns to analyze.")
    parser.add_argument("--prompt", type=str, default=defaults["prompt"], help="Classification prompt.")
    parser.add_argument("--choices", type=str, default=defaults["choices"], help="Comma-separated sentiment choices (e.g. positive,negative,neutral).")
    parser.add_argument("--sample", type=int, default=defaults["sample"], help="Sample size for dataset (default 1000).")
    parser.add_argument("--workers", type=int, default=defaults["workers"], help="Max workers for cpu threading (default 4).")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--full", action="store_true", help="Run full analysis (default is sample only).")

    args = parser.parse_args()

    # Prepare inputs
    input_path = args.file
    file_id = str(uuid.uuid4())
    filename = os.path.basename(input_path)
    upload_path = os.path.join(UPLOAD_DIR, f"{file_id}_{filename}")
    shutil.copy(input_path, upload_path)

    model = args.model
    workers = args.workers
    debug = args.debug

    columns = [col.strip() for col in args.columns.split(",")]
    choices = [c.strip() for c in args.choices.split(",")]

    try:
        df = pd.read_csv(upload_path)
    except Exception as e:
        print(f"❌ CSV parsing error: {e}")
        try:
            df = pd.read_excel(upload_path)
        except Exception as e:
            print(f"Excel parsing error: {e}")
            return

    if not all(col in df.columns for col in columns):
        print("❌ One or more selected columns do not exist in the CSV.")
        print("Available columns:", list(df.columns))
        return

    # Optional sampling
    if not args.full and args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42)

    df["__combined_text"] = df[columns].astype(str).agg(" ".join, axis=1)
    sentiments = analyse_sentiments(df["__combined_text"].tolist(), args.prompt, choices, model, max_workers=workers, debug=debug)
    df["Sentiment"] = sentiments

    output_path = os.path.join(RESULT_DIR, f"annotated_{filename}_{uuid.uuid4()}.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Sentiment analysis complete! Output saved to: {output_path}")


if __name__ == "__main__":
    main()