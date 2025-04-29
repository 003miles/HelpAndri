# PressScan
## Required Installations
- [Ollama](https://ollama.com/download)
- [Python](https://www.python.org/downloads/)

## Usage
Run the following command in your terminal
```
ollama pull mistral
```

Clone the repository
```
git clone https://github.com/003miles/PressScan.git
cd PressScan
```

Install the required packages
```
pip install -r requirements.txt
```

Run the script using
```
python cli_sentiment.py --file <CSV_FILE_PATH> [ARGUMENTS]
```

```
usage: cli_sentiment.py [-h] --file FILE [--model MODEL] [--columns COLUMNS] [--prompt PROMPT] [--choices CHOICES] [--sample SAMPLE] [--workers WORKERS] [--debug] [--full]
                        [--dryrun]

Run sentiment classification on a CSV using an LLM.

options:
  -h, --help         show this help message and exit
  --file FILE        Path to the input CSV file.
  --model MODEL      Model to use for sentiment classification.
  --columns COLUMNS  Comma-separated list of columns to analyze.
  --prompt PROMPT    Classification prompt (uses prompt.txt by default).
  --choices CHOICES  Comma-separated sentiment choices (e.g. positive,negative,neutral).
  --sample SAMPLE    Sample size for dataset (default 1000).
  --workers WORKERS  Max workers for cpu threading (default 4).
  --debug            Enable debug mode.
  --full             Run full analysis (default is sample only).
  --dryrun           Run without saving results.
```

The output CSV will be saved to the same folder
