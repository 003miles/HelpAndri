# PressScan 
<img width="1512" alt="HELPANDRI RUNNING crop" src="https://github.com/user-attachments/assets/8c903cd7-6eca-47c9-97ae-bebbe4cd8526" />

PressScan is a command-line Python tool for high-volume sentiment classification of `.csv` or `.xlsx` text datasets using a local large language model (LLM).

Built for large-scale projects such as news article analysis, where public LLM APIs would be too slow, expensive, or rate-limited.

[*Full documentation*](https://github.com/003miles/PressScan/wiki)
## Prerequisites
Before you begin, ensure you have the following installed
- [Ollama](https://ollama.com/download) - For running LLM models locally
- [Python 3.9 or above](https://www.python.org/downloads/) - Required to run the sentiment analyser scripts
### Why are these needed?
- Ollama manages local LLM serving (e.g, Mistral, LLaMA 3, Deepseek, etc.)
- Python 3.9+ ensures compatability with libraries like `pandas`
## Setup Instructions
### 1. Start the ollama server
Make sure the Ollama server is running locally to provide the LLM:
```bash
ollama serve
```
### 2. Pull the required model
Download a model to use (e.g., `mistral`):
```bash
ollama pull mistral
```
> To use another model, replace `mistral` with you chosen model name and pass it later using `--model`.
 
Example:
```bash
ollama pull llama3
```
### 3. Clone the repository
```bash
git clone https://github.com/003miles/PressScan.git
cd PressScan
```
### 4. Install the required packages
```bash
pip install -r requirements.txt
```
> Recommended to use a virtual environment (e.g., `python -m venv .venv`)
### 5. Run the Sentiment Analyser
```bash
python cli_sentiment.py --file <CSV_FILE_PATH> [OPTIONS]
```
Example:
```bash
python cli_sentiment.py --file dataset.csv --model mistral --columns title,quotes --choices positive,negative,neutral,unrelated --sample 1000 --workers 8 --debug
```
### Command Line Options
| Option      | Description                                               |
|:------------|:----------------------------------------------------------|
| `--file`    | (Required) Path to the input CSV file                     |
| `--model`   | Model name to use (default: `mistral`)                    |
| `--columns` | Comma-separated list of columns to combine for text input |
| `--prompt`  | Custom prompt text file (default: `prompt.txt`)           |
| `--choices` | Comma-separated list of classification labels             |
| `--sample`  | Sample size (default: 1000 rows)                          |
| `--workers` | Max workers for CPU threading (default: 4)                |
| `--debug`   | Output full LLM reasoning text along with answers         |
| `--full`    | Ignore sampling and run on the entire dataset             |
| `--dryrun`  | Process without saving the output CSV                     |
You can see help any time:
```bash
python cli_sentiment.py --help
```
### Output
The final annotated CSV file will be saved in the `temp_results/` folder.

Example:
```bash
temp_results/annotated_dataset.csv_899f4b62-0383-4f87-9d9d-e86b2249186c.csv
```
### Notes
- **Ollama** automatically manages GPU acceleration if available (especially useful for large datasets).
- Default prompt uses **reasoning first** then **final answer** to improve model accuracy (Chain-of-Thought prompting [Wei et al., 2022]).
- [RunPod](https://runpod.io/) or other cloud compute services can be used for faster batch processing.
