#!/usr/bin/env python3
"""
Hyperparameter Optimization Runner for DeepFlyer

Runs random search hyperparameter optimization with ClearML tracking.
Can be run standalone or integrated with training pipeline.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl_agent.algorithms.p3o import P3OConfig, HyperparameterOptimizer
from rl_agent.utils import ClearMLTracker


class HyperoptRunner:
    """Manages hyperparameter optimization trials"""
    
    def __init__(self, 
                 num_trials: int = 20,
                 episodes_per_trial: int = 100,
                 enable_clearml: bool = True):
        """
        Initialize hyperparameter optimization runner
        
        Args:
            num_trials: Number of random search trials
            episodes_per_trial: Episodes to evaluate each configuration
            enable_clearml: Whether to log to ClearML
        """
        self.num_trials = num_trials
        self.episodes_per_trial = episodes_per_trial
        
        # Initialize ClearML
        self.clearml_tracker = None
        if enable_clearml:
            try:
                self.clearml_tracker = ClearMLTracker(
                    project_name="DeepFlyer-Hyperopt",
                    task_name=f"Random Search {time.strftime('%Y%m%d-%H%M%S')}",
                    tags=['hyperopt', 'p3o', 'random-search']
                )
                print("ClearML tracking enabled for hyperparameter optimization")
            except Exception as e:
                print(f"Warning: ClearML initialization failed: {e}")
        
        # Initialize optimizer
        base_config = P3OConfig()
        self.optimizer = HyperparameterOptimizer(base_config, self.clearml_tracker)
        
        # Results storage
        self.results = []
        
    def evaluate_config(self, config: P3OConfig, trial_num: int) -> float:
        """
        Evaluate a hyperparameter configuration
        
        Args:
            config: P3O configuration to evaluate
            trial_num: Trial number
            
        Returns:
            Performance metric (average reward)
        """
        print(f"\nTrial {trial_num}/{self.num_trials}")
        print(f"Configuration:")
        print(f"  Learning rate: {config.learning_rate:.5f}")
        print(f"  Clip ratio: {config.clip_ratio:.3f}")
        print(f"  Entropy coef: {config.entropy_coef:.4f}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Rollout steps: {config.rollout_steps}")
        
        # Here you would normally run actual training
        # For now, simulate with a performance metric
        
        # Simulated performance (replace with actual training)
        base_performance = 100.0
        
        # Simulate that some hyperparameters work better
        lr_factor = 1.0 - abs(config.learning_rate - 3e-4) / 3e-4
        clip_factor = 1.0 - abs(config.clip_ratio - 0.2) / 0.2
        entropy_factor = 1.0 - abs(config.entropy_coef - 0.01) / 0.01
        
        performance = base_performance * (0.4 * lr_factor + 0.3 * clip_factor + 0.3 * entropy_factor)
        performance += np.random.normal(0, 10)  # Add noise
        
        print(f"  Performance: {performance:.2f}")
        
        return performance
    
    def run_optimization(self):
        """Run the full hyperparameter optimization"""
        print(f"Starting hyperparameter optimization with {self.num_trials} trials")
        print("=" * 60)
        
        for trial in range(self.num_trials):
            # Get suggested configuration
            config = self.optimizer.suggest_config()
            
            # Evaluate configuration
            performance = self.evaluate_config(config, trial + 1)
            
            # Report results
            self.optimizer.report_performance(
                config, 
                performance,
                additional_metrics={
                    'episodes': self.episodes_per_trial,
                    'trial_time': time.time()
                }
            )
            
            # Store results
            self.results.append({
                'trial': trial + 1,
                'config': config.__dict__,
                'performance': performance
            })
            
            # Update current trial
            self.optimizer.current_trial += 1
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
    
    def print_summary(self):
        """Print optimization summary"""
        print("\n" + "=" * 60)
        print("HYPERPARAMETER OPTIMIZATION COMPLETE")
        print("=" * 60)
        
        best_config = self.optimizer.get_best_config()
        if best_config:
            print(f"\nBest configuration (performance: {self.optimizer.best_performance:.2f}):")
            for key, value in best_config.__dict__.items():
                if not key.startswith('_'):
                    print(f"  {key}: {value}")
        
        # Top 3 trials
        sorted_results = sorted(self.results, key=lambda x: x['performance'], reverse=True)
        print("\nTop 3 trials:")
        for i, result in enumerate(sorted_results[:3]):
            print(f"  {i+1}. Trial {result['trial']}: {result['performance']:.2f}")
    
    def save_results(self):
        """Save optimization results to file"""
        results_dir = Path("hyperopt_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        results_file = results_dir / f"hyperopt_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'num_trials': self.num_trials,
                'episodes_per_trial': self.episodes_per_trial,
                'best_config': self.optimizer.best_config,
                'best_performance': self.optimizer.best_performance,
                'all_results': self.results
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Also save best config separately for easy loading
        if self.optimizer.best_config:
            best_config_file = results_dir / "best_config.json"
            with open(best_config_file, 'w') as f:
                json.dump(self.optimizer.best_config, f, indent=2)
            print(f"Best config saved to: {best_config_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization for DeepFlyer')
    parser.add_argument('--trials', type=int, default=20, help='Number of trials')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes per trial')
    parser.add_argument('--no-clearml', action='store_true', help='Disable ClearML tracking')
    
    args = parser.parse_args()
    
    runner = HyperoptRunner(
        num_trials=args.trials,
        episodes_per_trial=args.episodes,
        enable_clearml=not args.no_clearml
    )
    
    try:
        runner.run_optimization()
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
        runner.print_summary()
        runner.save_results()


if __name__ == "__main__":
    main()
