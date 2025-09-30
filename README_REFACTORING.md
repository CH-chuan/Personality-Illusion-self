# Behavioral Tasks Refactoring Guide

## Structure

The behavioral tasks have been refactored from Jupyter notebooks to modular Python files:

```
Personality-Illusion-self/
├── api_utils.py                           # Shared API utilities (from notebook cells 3-5)
├── run_behavioral_tasks.py                # Main runner with experimental loop
└── behavioral_tasks/
    ├── confidence_calibration.py          # Honesty task (from Honesty.ipynb cell 5)
    ├── iat.py                             # IAT task (from IAT.ipynb cell 6)
    ├── risk_taking.py                     # CCT risk-taking (from RiskTaking.ipynb cell 5)
    └── sycophancy.py                      # Moral sycophancy (from Sycophancy.ipynb cell 5)
```

## Code Reuse Mapping

### Shared Components
- **api_utils.py** reuses:
  - `initialize_clients()` - from all notebooks cell 3
  - `call_*_api()` functions - from all notebooks cell 4
  - `route_api_call()` - from all notebooks cell 4
  - `process_batch_requests()` - from all notebooks cell 4

### Task Modules
Each task module (`confidence_calibration.py`, `iat.py`, `risk_taking.py`, `sycophancy.py`) reuses:
- Core experiment class from original notebook cell 5/6
- All prompt building methods
- Response parsing methods
- Metrics calculation methods
- Added convenience function `run_*_task()` for easier usage

## Setup

1. **Set API keys as environment variables:**
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export TOGETHER_API_KEY="your-key-here"
export OPENROUTER_API_KEY="your-key-here"
```

2. **Install dependencies:**
```bash
pip install anthropic openai together backoff fuzzywuzzy python-levenshtein pandas numpy
```

3. **Prepare data files:**
- `behavioral_tasks/norm300.csv` - Trivia questions for confidence task
- `behavioral_tasks/iat_stimuli.json` - IAT test stimuli
- `behavioral_tasks/dilemmas.json` - Moral dilemmas for sycophancy task

## Usage

### Option 1: Run All Tasks (Full Experimental Loop with Auto-Resume)

```python
from run_behavioral_tasks import run_experimental_loop

model_config = {
    "name": "gpt-4o",
    "provider": "openai",
    "max_tokens": 256
}

# Run all tasks with personas × temperatures × runs
# Features:
# - Saves after each configuration (minimal viable level)
# - Auto-resumes if interrupted (Ctrl+C or error)
# - Skips already completed configurations
results = run_experimental_loop(
    task_names=['confidence', 'iat', 'risk_taking', 'sycophancy'],
    model_config=model_config,
    temperatures=[0.3, 0.7, 1.0],
    num_runs=3,
    resume=True  # Default: True, will resume from checkpoint
)
```

### Option 2: Run Individual Tasks

```python
from behavioral_tasks.confidence_calibration import run_confidence_task
from behavioral_tasks.iat import run_iat_task
from behavioral_tasks.risk_taking import run_risk_taking_task
from behavioral_tasks.sycophancy import run_sycophancy_task

model_config = {
    "name": "gpt-4o",
    "provider": "openai",
    "temperature": 0.7,
    "max_tokens": 256
}

# Run individual task
summary, raw_results = run_confidence_task(
    model_config,
    persona_content="You are a helpful assistant."
)
```

### Option 3: Use Experiment Classes Directly

```python
from behavioral_tasks.confidence_calibration import ConfidenceCalibrationExperiment

experiment = ConfidenceCalibrationExperiment(
    questions_file="behavioral_tasks/norm300.csv",
    questions_per_bin=10
)

results_df = experiment.run_experiment_api(
    model_config,
    persona_content="",
    max_workers=8
)
```

## Integrating with Your Personality Results

Since you already have Big Five personality results, you can use them to configure personas:

```python
import pandas as pd

# Load your personality results
personality_df = pd.read_csv("your_personality_results.csv")

# Create persona based on personality traits
high_agreeable = personality_df[personality_df['Agreeableness'] > 4.0].iloc[0]
persona_content = f"You are someone who scores high on agreeableness (score: {high_agreeable['Agreeableness']})."

# Run behavioral tasks with this persona
summary, results = run_iat_task(
    model_config,
    persona_content=persona_content
)
```

## Experimental Loop (Replicated from Notebooks)

The `run_experimental_loop()` function replicates the experimental design from notebooks:

**From notebooks (Cells 6-7):**
- Loops over: personas × temperatures × runs
- Runs all tasks for each configuration
- Saves summary and raw results
- Adds delays between API calls

**Key parameters:**
- `personas`: Dict of persona configurations (default: 3 baseline personas)
- `temperatures`: List of temperatures (default: [0.3, 0.7, 1.0])
- `num_runs`: Number of runs per config (default: 3)
- `max_workers`: Parallel API calls (default: 8)
- `batch_delay`: Seconds between configs (default: 2.0)

## Output

Results are saved to `behavioral_results/` (or custom `output_dir`):

### Incremental Saves (during experiment):
- `{model}_summary_incremental.csv` - Summary updated after each config
- `raw_results/{model}_{task}_p{persona}_t{temp}_r{run}_{timestamp}.csv` - Individual raw results
- `.checkpoint_{model}.json` - Hidden checkpoint file for resume

### Final Saves (on completion):
- `{model}_summary_final_{timestamp}.csv` - Final summary
- `{model}_{task}_raw_final_{timestamp}.csv` - Consolidated raw results per task

## Robustness Features

### Incremental Saving
- **Saves after each configuration** (persona × temperature × run)
- **Raw results saved immediately** after each task completes
- **Summary updated continuously** - never lose progress
- **Minimum viable level**: Each config is the smallest atomic unit

### Pause and Resume
- **Checkpoint system**: Tracks completed configurations
- **Auto-resume**: Run script again with `resume=True` (default)
- **Skips completed**: Won't re-run already finished configs
- **Handles interruptions**: Ctrl+C or unexpected errors
- **Clear on success**: Checkpoint auto-deleted when experiment completes

### Example Workflow

```bash
# Start experiment
python run_behavioral_tasks.py

# ... experiment runs for a while ...
# Press Ctrl+C or encounter error

# Output:
# ⚠️  INTERRUPTED by user!
# Progress saved. Completed 15/27 configurations.
# To resume, run the script again with resume=True (default)

# Resume (will skip the 15 completed configs)
python run_behavioral_tasks.py

# Output:
# 📂 Loaded checkpoint: 15 configs already completed
# [16/27] ⏭️  SKIPPED (already completed): Persona 1 | Temp 0.7 | Run 1
# [17/27] Persona 2 | Temp 0.7 | Run 1
# ... continues from where it left off ...
```

## Task-Specific Notes

### Confidence Calibration (Honesty)
- Requires `norm300.csv` with trivia questions
- Two-step process: answer + confidence, then re-rate confidence
- Metrics: accuracy, calibration (ECE), Brier score

### IAT (Implicit Bias)
- Requires `iat_stimuli.json` with category/attribute pairs
- Generates 3 random orders per test category
- Metrics: bias scores per category

### Risk-Taking (CCT)
- 27 scenarios (3×3×3 factorial design)
- Metrics: average cards chosen, risk aversion, optimal choice percentage

### Sycophancy
- Requires `dilemmas.json` with moral dilemmas
- Two-step: baseline answer, then with opposing suggestion
- Metrics: percentage of changed answers (sycophancy rate)

## Differences from Notebooks

1. **API key handling**: Uses environment variables instead of Google Colab secrets
2. **Module structure**: Separated into importable modules
3. **Convenience functions**: Added `run_*_task()` for easier single-task usage
4. **Cleaner imports**: No need for sys.path manipulation in notebooks

## Example: Quick Test

```bash
cd /Users/chuan/Desktop/projects/Personality-Illusion-self
python run_behavioral_tasks.py
```

This runs a quick test with one temperature and one run.