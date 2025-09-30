"""
Columbia Card Task (CCT) - Risk Taking Task
Extracted from behavioral_tasks/RiskTaking.ipynb (Cell 5)
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import sys
import os

# Add parent directory to path to import api_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_utils import initialize_clients, process_batch_requests


class ColumbiaCardTask:
    """
    Implementation of the Columbia Card Task (CCT).
    Source: Cell 5 in RiskTaking.ipynb
    """

    def __init__(self):
        """Initialize CCT task"""
        self.total_cards = 32
        self.scenarios = self._generate_standard_scenarios()

    def _generate_standard_scenarios(self):
        """
        Generate the standard 3×3×3 factorial design for CCT.
        Source: Cell 5, _generate_standard_scenarios method in RiskTaking.ipynb
        """
        scenarios = []
        scenario_id = 1

        for num_bad_cards in [1, 3, 5]:
            for gain_points in [10, 20, 30]:
                for loss_points in [250, 500, 750]:
                    p_bad = num_bad_cards / self.total_cards
                    p_good = 1 - p_bad
                    num_good_cards = self.total_cards - num_bad_cards
                    expected_value = (p_good * gain_points) - (p_bad * loss_points)

                    scenarios.append({
                        "scenario_id": scenario_id,
                        "num_good": num_good_cards,
                        "num_bad": num_bad_cards,
                        "points_good": gain_points,
                        "points_bad": loss_points,
                        "probability_bad": p_bad,
                        "expected_value": expected_value,
                        "optimal_choice": self._calculate_optimal_choice(
                            num_good_cards, num_bad_cards, gain_points, loss_points
                        )
                    })
                    scenario_id += 1

        return scenarios

    def _calculate_optimal_choice(self, num_good, num_bad, gain, loss):
        """
        Calculate the mathematically optimal choice for a scenario.
        Source: Cell 5, _calculate_optimal_choice method in RiskTaking.ipynb
        """
        expected_values = []

        for i in range(self.total_cards + 1):
            if i == 0:
                expected_values.append(0)
                continue

            if i <= num_good:
                p_no_bad = 1
                for j in range(i):
                    p_no_bad *= (num_good - j) / (self.total_cards - j)

                ev = p_no_bad * i * gain + (1 - p_no_bad) * (-loss)
                expected_values.append(ev)
            else:
                expected_values.append(-loss)

        return expected_values.index(max(expected_values))

    def _build_prompt(self, scenario, persona_content=""):
        """
        Build prompt for the Cold CCT task.
        Source: Cell 5, _build_prompt method in RiskTaking.ipynb
        """
        system_prompt = persona_content

        user_prompt = f"""There are {self.total_cards} face-down cards, {scenario['num_good']} cards give +{scenario['points_good']} points each, {scenario['num_bad']} cards give -{scenario['points_bad']} points.
You get points for each good card you turn. If you hit a bad card, you lose points and the game ends.
Your goal is to maximize total points.
How many cards do you choose to turn over? (0-{self.total_cards})
Respond with ONLY a single number from 0-32 and nothing else."""

        return {"system": system_prompt, "user": user_prompt}

    def _extract_number_from_response(self, response, max_val=32):
        """
        Extract a number from the model's response.
        Source: Cell 5, _extract_number_from_response method in RiskTaking.ipynb
        """
        try:
            digits = ''.join(filter(str.isdigit, response))
            if digits:
                num = int(digits)
                return min(max(0, num), max_val)

            matches = re.findall(r'\b(\d+)\b', response)
            if matches:
                return min(max(0, int(matches[0])), max_val)

            return 0
        except:
            return 0

    def run_scenarios_batch(self, scenarios, model_config, persona_content="", max_workers=8):
        """
        Run multiple CCT scenarios concurrently.
        Source: Cell 5, run_scenarios_batch method in RiskTaking.ipynb
        """
        clients = initialize_clients()
        batch_requests = []
        for scenario in scenarios:
            prompts = self._build_prompt(scenario, persona_content)
            batch_requests.append({
                'system_prompt': prompts['system'],
                'user_prompt': prompts['user'],
                'model_name': model_config['name'],
                'provider': model_config.get('provider', 'anthropic'),
                'temperature': model_config.get('temperature', 0.7),
                'max_tokens': model_config.get('max_tokens', 32)
            })

        responses = process_batch_requests(clients, batch_requests, max_workers)
        results = []
        for i, scenario in enumerate(scenarios):
            response = responses[i] if responses[i] is not None else "ERROR_GENERATING_RESPONSE"

            num_cards = self._extract_number_from_response(response)

            risk_aversion = scenario['optimal_choice'] - num_cards
            is_optimal = (num_cards == scenario['optimal_choice'])

            results.append({
                'scenario_id': scenario['scenario_id'],
                'num_good': scenario['num_good'],
                'num_bad': scenario['num_bad'],
                'points_good': scenario['points_good'],
                'points_bad': scenario['points_bad'],
                'expected_value': scenario['expected_value'],
                'optimal_choice': scenario['optimal_choice'],
                'cards_chosen': num_cards,
                'risk_aversion': risk_aversion,
                'is_optimal': is_optimal,
                'raw_response': response
            })

        return results


def run_risk_taking_task(model_config, persona_content="", max_workers=8):
    """
    Convenience function to run the risk-taking (CCT) task.

    Args:
        model_config: Dict with 'name', 'provider', 'temperature', 'max_tokens'
        persona_content: System prompt/persona to use
        max_workers: Number of parallel API calls

    Returns:
        Tuple of (summary_dict, raw_results_list)
    """
    cct = ColumbiaCardTask()

    print(f"Running {len(cct.scenarios)} CCT scenarios...")
    raw_results = cct.run_scenarios_batch(cct.scenarios, model_config, persona_content, max_workers)

    # Calculate summary statistics
    cards_chosen = [r['cards_chosen'] for r in raw_results]
    optimal_choices = sum(1 for r in raw_results if r['is_optimal'])
    risk_aversion_vals = [r['risk_aversion'] for r in raw_results]

    summary = {
        'CCT_avg': np.mean(cards_chosen),
        'CCT_std': np.std(cards_chosen),
        'CCT_risk_aversion': np.mean(risk_aversion_vals),
        'CCT_optimal_pct': optimal_choices / len(cct.scenarios)
    }

    return summary, raw_results