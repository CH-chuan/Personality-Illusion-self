# Setup Guide for LiteLLM/vLLM Experiment

This guide will help you set up and run behavioral tasks with your `hosted_vllm/Qwen3-Coder-30B-A3B-Instruct` model via LiteLLM.

## Prerequisites

1. **Python packages** (already installed if you followed README_REFACTORING.md):
```bash
pip install anthropic openai together backoff fuzzywuzzy python-levenshtein pandas numpy python-dotenv
```

2. **Data files** (should already be in place):
- `behavioral_tasks/norm300.csv` - Trivia questions
- `behavioral_tasks/iat_stimuli.json` - IAT stimuli
- `behavioral_tasks/dilemmas.json` - Moral dilemmas

## Step 1: Configure Environment Variables

1. **Copy the example file**:
```bash
cp .env.example .env
```

2. **Edit `.env` file** with your LiteLLM credentials:
```bash
# Only need these two for LiteLLM
LITELLM_API_KEY=your_actual_litellm_api_key_here
LITELLM_BASE_URL=https://your-vllm-endpoint.com/v1
```

**Note**: The `.env` file is gitignored, so your API keys stay private.

## Step 2: Customize Personality Prompts (Optional)

Open `main.py` and find the `PERSONAS` section (lines 25-88):

```python
PERSONAS = {
    "experimental": [
        {
            "index": 0,
            "name": "no_personality",
            "content": ""  # Baseline - no personality
        },
        {
            "index": 1,
            "name": "high_openness",
            "content": "You are creative, curious, and open to new experiences..."
            # TODO: CUSTOMIZE THIS PROMPT
        },
        # ... more personas
    ]
}
```

**Customize each persona prompt** to match your research design. The default prompts are provided as examples.

## Step 3: Verify Configuration

Current settings in `main.py`:
- **Model**: `hosted_vllm/Qwen3-Coder-30B-A3B-Instruct`
- **Provider**: `litellm`
- **Temperature**: `0.7`
- **Runs per persona**: `10`
- **Tasks**: All 4 (confidence, IAT, risk-taking, sycophancy)
- **Personas**: 11 (no personality + Big Five high/low)
- **Total configs**: `11 personas × 10 runs = 110 configurations`

## Step 4: Run the Experiment

```bash
python main.py
```

You'll see a confirmation prompt:
```
======================================================================
BEHAVIORAL TASKS EXPERIMENT
======================================================================
Model: hosted_vllm/Qwen3-Coder-30B-A3B-Instruct
Provider: litellm
Temperature: 0.7
Runs per persona: 10
Tasks: confidence, iat, risk_taking, sycophancy
Personas: 11 (no personality + Big Five)
Total configurations: 110
======================================================================

Ready to start experiment? This will take a while. (y/n):
```

Type `y` to start.

## Step 5: Monitor Progress

The experiment will:
1. **Save incrementally** after each configuration
2. **Show progress**: `[15/110] Persona 3 | Temp 0.7 | Run 5`
3. **Save raw results** immediately after each task

Output files:
```
behavioral_results/
├── .checkpoint_hosted_vllm_Qwen3-Coder-30B-A3B-Instruct.json
├── hosted_vllm_Qwen3-Coder-30B-A3B-Instruct_summary_incremental.csv
└── raw_results/
    ├── hosted_vllm_Qwen3-Coder-30B-A3B-Instruct_confidence_p0_t0.7_r1_*.csv
    ├── hosted_vllm_Qwen3-Coder-30B-A3B-Instruct_iat_p0_t0.7_r1_*.csv
    └── ...
```

## Step 6: Pause and Resume (If Needed)

If you need to stop (Ctrl+C) or encounter an error:

```bash
⚠️  INTERRUPTED by user!
Progress saved. Completed 45/110 configurations.
To resume, run the script again with resume=True (default)
```

Just run again:
```bash
python main.py
```

It will automatically:
- ✅ Load checkpoint
- ✅ Skip completed configs (45 in this case)
- ✅ Continue from config 46

## Estimated Time

Rough estimates (depends on API speed):
- **Per task**: ~30-60 seconds
- **Per config** (4 tasks): ~2-4 minutes
- **Full experiment** (110 configs): ~4-7 hours

With 10 runs per persona, you'll get good statistical power for your analysis.

## Troubleshooting

### Issue: LiteLLM client not initialized
```
✗ Failed to initialize LiteLLM client
```
**Solution**: Check your `.env` file:
- Verify `LITELLM_API_KEY` is set correctly
- Verify `LITELLM_BASE_URL` points to your vLLM endpoint

### Issue: Connection errors
**Solution**:
- Test your endpoint: `curl -X POST $LITELLM_BASE_URL/chat/completions`
- Check if vLLM server is running
- Verify API key permissions

### Issue: Rate limiting
**Solution**: Adjust `MAX_WORKERS` in `main.py` (line 114):
```python
MAX_WORKERS = 4  # Reduce from 8 to 4 for slower rate
```

### Issue: Model name mismatch
**Solution**: Update model name in `main.py` (line 15) to match your LiteLLM configuration:
```python
"name": "your-actual-model-name",  # Must match LiteLLM proxy config
```

## After Completion

Final files will be created:
```
behavioral_results/
├── hosted_vllm_Qwen3-Coder-30B-A3B-Instruct_summary_final_20250930_120000.csv
├── hosted_vllm_Qwen3-Coder-30B-A3B-Instruct_confidence_raw_final_*.csv
├── hosted_vllm_Qwen3-Coder-30B-A3B-Instruct_iat_raw_final_*.csv
├── hosted_vllm_Qwen3-Coder-30B-A3B-Instruct_risk_taking_raw_final_*.csv
└── hosted_vllm_Qwen3-Coder-30B-A3B-Instruct_sycophancy_raw_final_*.csv
```

The checkpoint file will be automatically deleted on successful completion.

## Next Steps

After data collection:
1. Analyze summary CSV for overall patterns
2. Examine raw results for detailed behavior
3. Compare no-personality baseline with Big Five personas
4. Look for dissociations between self-reports and behaviors (if you ran personality tests separately)