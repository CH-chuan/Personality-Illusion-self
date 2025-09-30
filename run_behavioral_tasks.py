"""
Main runner for behavioral tasks with incremental saving and pause/resume.
Handles the experimental loop: personas × temperatures × runs
"""

import pandas as pd
import time
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

# Import task modules
from behavioral_tasks.confidence_calibration import run_confidence_task, ConfidenceCalibrationExperiment
from behavioral_tasks.iat import run_iat_task
from behavioral_tasks.risk_taking import run_risk_taking_task
from behavioral_tasks.sycophancy import run_sycophancy_task


# =========================================================================
# Default Configuration
# =========================================================================

DEFAULT_PERSONAS = {
    "baseline": [
        {"index": 1, "content": ""},
        {"index": 2, "content": "You are a helpful assistant."},
        {"index": 3, "content": "Respond to the instructions."}
    ]
}

DEFAULT_TEMPERATURES = [0.3, 0.7, 1.0]
DEFAULT_NUM_RUNS = 3
DEFAULT_MAX_WORKERS = 8
DEFAULT_BATCH_DELAY = 2.0  # seconds between configurations


# =========================================================================
# Checkpoint Management
# =========================================================================

def save_checkpoint(output_dir: str, model_name: str, completed_configs: List[Dict]):
    """Save checkpoint of completed configurations."""
    checkpoint_file = f"{output_dir}/.checkpoint_{model_name.replace('/', '_')}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'completed_configs': completed_configs,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"  💾 Checkpoint saved: {len(completed_configs)} configs completed")


def load_checkpoint(output_dir: str, model_name: str) -> List[Dict]:
    """Load checkpoint of completed configurations."""
    checkpoint_file = f"{output_dir}/.checkpoint_{model_name.replace('/', '_')}.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            completed = data['completed_configs']
            print(f"📂 Loaded checkpoint: {len(completed)} configs already completed")
            return completed
    return []


def is_config_completed(persona_index: int, temperature: float, run: int, completed_configs: List[Dict]) -> bool:
    """Check if a configuration has already been completed."""
    for config in completed_configs:
        if (config['persona_index'] == persona_index and
            config['temperature'] == temperature and
            config['run'] == run):
            return True
    return False


def clear_checkpoint(output_dir: str, model_name: str):
    """Clear checkpoint file after successful completion."""
    checkpoint_file = f"{output_dir}/.checkpoint_{model_name.replace('/', '_')}.json"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"✓ Checkpoint cleared")


# =========================================================================
# Incremental Saving
# =========================================================================

def save_incremental_summary(output_dir: str, model_name: str, all_summary_results: List[Dict]):
    """Save summary incrementally after each configuration."""
    model_name_clean = model_name.replace('/', '_')
    summary_filename = f"{output_dir}/{model_name_clean}_summary_incremental.csv"

    summary_df = pd.DataFrame(all_summary_results)
    summary_df.to_csv(summary_filename, index=False)
    return summary_filename


def save_incremental_raw(output_dir: str, model_name: str, task_name: str,
                         persona_index: int, temperature: float, run: int,
                         raw_results):
    """Save raw results incrementally after each task completion."""
    model_name_clean = model_name.replace('/', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Create subdirectory for raw results
    raw_dir = f"{output_dir}/raw_results"
    os.makedirs(raw_dir, exist_ok=True)

    # Save with unique filename including config details
    filename = f"{raw_dir}/{model_name_clean}_{task_name}_p{persona_index}_t{temperature}_r{run}_{timestamp}.csv"

    if isinstance(raw_results, pd.DataFrame):
        raw_results.to_csv(filename, index=False)
    elif isinstance(raw_results, list):
        pd.DataFrame(raw_results).to_csv(filename, index=False)

    return filename


# =========================================================================
# Task Runner Functions
# =========================================================================

def run_single_task(task_name, model_config, persona_content, task_kwargs=None):
    """
    Run a single behavioral task.

    Args:
        task_name: One of 'confidence', 'iat', 'risk_taking', 'sycophancy'
        model_config: Dict with 'name', 'provider', 'temperature', 'max_tokens'
        persona_content: System prompt/persona string
        task_kwargs: Additional kwargs for specific tasks

    Returns:
        Tuple of (summary_dict, raw_results)
    """
    task_kwargs = task_kwargs or {}

    if task_name == 'confidence':
        experiment = ConfidenceCalibrationExperiment(
            questions_file=task_kwargs.get('questions_file', 'norm300.csv'),
            questions_per_bin=task_kwargs.get('questions_per_bin', 10)
        )
        results_df = experiment.run_experiment_api(
            model_config,
            persona_content,
            max_workers=task_kwargs.get('max_workers', 8)
        )

        # Calculate summary metrics
        avg_accuracy_em = results_df['is_correct_em'].mean() * 100
        avg_confidence_c1 = results_df['confidence_c1'].mean()
        avg_overconfidence = (results_df['confidence_c1']/100 - results_df['is_correct_em']).mean() * 100
        overall_ece, overall_brier, bin_metrics = experiment.calculate_calibration_metrics(results_df)

        summary = {
            'accuracy_em': avg_accuracy_em,
            'confidence_c1': avg_confidence_c1,
            'overconfidence': avg_overconfidence,
            'ece': overall_ece * 100,
            'brier': overall_brier
        }
        summary.update(bin_metrics)

        return summary, results_df

    elif task_name == 'iat':
        return run_iat_task(
            model_config,
            persona_content,
            iat_json_path=task_kwargs.get('iat_json_path', 'iat_stimuli.json'),
            max_workers=task_kwargs.get('max_workers', 8)
        )

    elif task_name == 'risk_taking':
        return run_risk_taking_task(
            model_config,
            persona_content,
            max_workers=task_kwargs.get('max_workers', 8)
        )

    elif task_name == 'sycophancy':
        return run_sycophancy_task(
            model_config,
            persona_content,
            dilemmas_file=task_kwargs.get('dilemmas_file', 'dilemmas.json'),
            max_workers=task_kwargs.get('max_workers', 8)
        )

    else:
        raise ValueError(f"Unknown task: {task_name}. Choose from: confidence, iat, risk_taking, sycophancy")


def run_experimental_loop(
    task_names: List[str],
    model_config: Dict,
    personas: Dict = None,
    temperatures: List[float] = None,
    num_runs: int = None,
    max_workers: int = None,
    batch_delay: float = None,
    task_kwargs: Dict = None,
    output_dir: str = "behavioral_results",
    resume: bool = True
):
    """
    Run behavioral tasks across experimental conditions with incremental saving.

    Features:
    - Saves after each configuration (persona × temperature × run)
    - Can resume from checkpoint if interrupted
    - Saves raw results immediately after each task

    Args:
        task_names: List of task names to run (e.g., ['confidence', 'iat'])
        model_config: Dict with 'name', 'provider', 'max_tokens'
        personas: Dict of persona configurations (default: DEFAULT_PERSONAS)
        temperatures: List of temperature values (default: [0.3, 0.7, 1.0])
        num_runs: Number of runs per configuration (default: 3)
        max_workers: Number of parallel API calls (default: 8)
        batch_delay: Delay between configurations in seconds (default: 2.0)
        task_kwargs: Dict of task-specific kwargs
        output_dir: Directory to save results
        resume: Whether to resume from checkpoint (default: True)

    Returns:
        DataFrame with all results
    """
    # Use defaults if not specified
    personas = personas or DEFAULT_PERSONAS
    temperatures = temperatures or DEFAULT_TEMPERATURES
    num_runs = num_runs or DEFAULT_NUM_RUNS
    max_workers = max_workers or DEFAULT_MAX_WORKERS
    batch_delay = batch_delay or DEFAULT_BATCH_DELAY
    task_kwargs = task_kwargs or {}

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    model_name = model_config['name']
    model_name_clean = model_name.replace('/', '_')

    # Load checkpoint if resuming
    completed_configs = []
    if resume:
        completed_configs = load_checkpoint(output_dir, model_name)

    # Load existing summary if it exists
    summary_filename = f"{output_dir}/{model_name_clean}_summary_incremental.csv"
    if os.path.exists(summary_filename) and resume:
        all_summary_results = pd.read_csv(summary_filename).to_dict('records')
        print(f"📂 Loaded existing summary: {len(all_summary_results)} results")
    else:
        all_summary_results = []

    print(f"\n{'='*60}")
    print(f"Running Behavioral Tasks for {model_name}")
    print(f"{'='*60}")
    print(f"Tasks: {', '.join(task_names)}")
    print(f"Personas: {len(personas['baseline'])}")
    print(f"Temperatures: {temperatures}")
    print(f"Runs per configuration: {num_runs}")
    print(f"Resume mode: {'ON' if resume else 'OFF'}")

    total_configs = len(personas['baseline']) * len(temperatures) * num_runs
    config_count = 0
    skipped_count = 0

    # Main experimental loop: personas × temperatures × runs
    try:
        for persona_data in personas['baseline']:
            persona_index = persona_data['index']
            persona_content = persona_data['content']

            for temperature in temperatures:
                for run in range(1, num_runs + 1):
                    config_count += 1

                    # Check if this config was already completed
                    if is_config_completed(persona_index, temperature, run, completed_configs):
                        skipped_count += 1
                        print(f"\n[{config_count}/{total_configs}] ⏭️  SKIPPED (already completed): Persona {persona_index} | Temp {temperature} | Run {run}")
                        continue

                    print(f"\n[{config_count}/{total_configs}] Persona {persona_index} | Temp {temperature} | Run {run}")

                    # Prepare model config for this run
                    model_config_run = model_config.copy()
                    model_config_run['temperature'] = temperature

                    # Run each task
                    config_summary = {
                        'model': model_name,
                        'provider': model_config['provider'],
                        'persona_index': persona_index,
                        'persona_content': persona_content,
                        'temperature': temperature,
                        'run': run,
                        'timestamp': datetime.now().isoformat()
                    }

                    config_successful = True

                    for task_name in task_names:
                        print(f"  Running {task_name} task...")
                        try:
                            task_specific_kwargs = task_kwargs.get(task_name, {})
                            task_specific_kwargs['max_workers'] = max_workers

                            summary, raw_results = run_single_task(
                                task_name,
                                model_config_run,
                                persona_content,
                                task_specific_kwargs
                            )

                            # Add summary to config
                            for key, value in summary.items():
                                config_summary[f'{task_name}_{key}'] = value

                            # Save raw results immediately
                            raw_filename = save_incremental_raw(
                                output_dir, model_name, task_name,
                                persona_index, temperature, run,
                                raw_results
                            )

                            print(f"    ✓ Completed {task_name}")
                            print(f"    💾 Raw results saved: {os.path.basename(raw_filename)}")

                        except Exception as e:
                            print(f"    ✗ Error in {task_name}: {e}")
                            config_successful = False
                            config_summary[f'{task_name}_error'] = str(e)

                    # Add config to summary
                    all_summary_results.append(config_summary)

                    # Save summary incrementally
                    summary_file = save_incremental_summary(output_dir, model_name, all_summary_results)
                    print(f"  💾 Summary updated: {os.path.basename(summary_file)}")

                    # Mark as completed if successful
                    if config_successful:
                        completed_configs.append({
                            'persona_index': persona_index,
                            'temperature': temperature,
                            'run': run
                        })
                        save_checkpoint(output_dir, model_name, completed_configs)

                    # Delay between configurations
                    time.sleep(batch_delay)

    except KeyboardInterrupt:
        print(f"\n\n⚠️  INTERRUPTED by user!")
        print(f"Progress saved. Completed {len(completed_configs)}/{total_configs} configurations.")
        print(f"To resume, run the script again with resume=True (default)")

    except Exception as e:
        print(f"\n\n⚠️  ERROR: {e}")
        print(f"Progress saved. Completed {len(completed_configs)}/{total_configs} configurations.")
        print(f"To resume, run the script again with resume=True (default)")
        raise

    # Final consolidation
    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"{'='*60}")
    print(f"Total configurations: {total_configs}")
    print(f"Completed: {len(completed_configs)}")
    print(f"Skipped (already done): {skipped_count}")

    # Create final consolidated files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save final summary
    summary_df = pd.DataFrame(all_summary_results)
    final_summary = f"{output_dir}/{model_name_clean}_summary_final_{timestamp}.csv"
    summary_df.to_csv(final_summary, index=False)
    print(f"\n✓ Final summary: {final_summary}")

    # Consolidate raw results by task
    raw_dir = f"{output_dir}/raw_results"
    if os.path.exists(raw_dir):
        for task_name in task_names:
            task_files = [f for f in os.listdir(raw_dir) if f.startswith(f"{model_name_clean}_{task_name}_")]
            if task_files:
                dfs = []
                for f in task_files:
                    try:
                        df = pd.read_csv(os.path.join(raw_dir, f))
                        dfs.append(df)
                    except:
                        pass

                if dfs:
                    consolidated_df = pd.concat(dfs, ignore_index=True)
                    final_raw = f"{output_dir}/{model_name_clean}_{task_name}_raw_final_{timestamp}.csv"
                    consolidated_df.to_csv(final_raw, index=False)
                    print(f"✓ {task_name} consolidated: {final_raw}")

    # Clear checkpoint
    clear_checkpoint(output_dir, model_name)

    return summary_df


# =========================================================================
# Example Usage
# =========================================================================

if __name__ == "__main__":
    # Example: Run all tasks for one model

    model_config = {
        "name": "gpt-4o",
        "provider": "openai",
        "max_tokens": 256  # Use 256 for IAT/Sycophancy, 32 for others is fine
    }

    # Task-specific configurations
    task_kwargs = {
        'confidence': {
            'questions_file': 'behavioral_tasks/norm300.csv',
            'questions_per_bin': 10
        },
        'iat': {
            'iat_json_path': 'behavioral_tasks/iat_stimuli.json'
        },
        'sycophancy': {
            'dilemmas_file': 'behavioral_tasks/dilemmas.json'
        }
    }

    # Run all tasks
    results = run_experimental_loop(
        task_names=['confidence', 'iat', 'risk_taking', 'sycophancy'],
        model_config=model_config,
        personas=DEFAULT_PERSONAS,
        temperatures=[0.7],  # Quick test with one temperature
        num_runs=1,  # Quick test with one run
        task_kwargs=task_kwargs,
        resume=True  # Will resume from checkpoint if interrupted
    )

    print("\nResults preview:")
    print(results.head())