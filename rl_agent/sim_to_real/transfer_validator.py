"""
Sim-to-Real Transfer Validation for DeepFlyer

Validates model performance transfer from simulation to real-world conditions
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TransferMetrics:
    """Container for sim-to-real transfer metrics"""
    success_rate: float
    navigation_precision: float
    safety_violations: float
    average_completion_time: float
    control_smoothness: float
    energy_efficiency: float
    robustness_score: float


class SimToRealValidator:
    """Validates model performance in real-world conditions"""
    
    def __init__(self):
        """Initialize sim-to-real validator"""
        self.sim_metrics = {}
        self.real_metrics = {}
        
        # Define acceptable transfer gaps
        self.transfer_gap_thresholds = {
            'success_rate_gap': 0.15,      # 15% max drop in success rate
            'precision_gap': 0.10,         # 10% max drop in precision
            'safety_gap': 0.05,            # 5% max increase in safety violations
            'efficiency_gap': 0.20,        # 20% max drop in efficiency
            'control_gap': 0.15            # 15% max drop in control quality
        }
        
        # Transfer quality assessment weights
        self.quality_weights = {
            'success_rate': 0.30,
            'safety': 0.25,
            'precision': 0.20,
            'efficiency': 0.15,
            'control_quality': 0.10
        }
        
    def validate_transfer(self, sim_results: Dict[str, Any], real_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate sim-to-real transfer quality"""
        logger.info("Validating sim-to-real transfer performance")
        
        # Extract metrics from results
        sim_metrics = self._extract_metrics(sim_results, 'simulation')
        real_metrics = self._extract_metrics(real_results, 'real_world')
        
        # Calculate transfer gaps
        transfer_gaps = self._calculate_transfer_gaps(sim_metrics, real_metrics)
        
        # Assess transfer quality
        transfer_quality = self._calculate_transfer_quality(transfer_gaps)
        
        # Analyze specific transfer challenges
        challenge_analysis = self._analyze_transfer_challenges(transfer_gaps, sim_metrics, real_metrics)
        
        # Generate adaptation recommendations
        adaptation_recommendations = self._generate_adaptation_recommendations(transfer_gaps, challenge_analysis)
        
        # Determine transfer success
        transfer_successful = self._assess_transfer_success(transfer_quality, transfer_gaps)
        
        return {
            'transfer_quality': transfer_quality,
            'transfer_gaps': transfer_gaps,
            'transfer_successful': transfer_successful,
            'sim_metrics': sim_metrics.__dict__,
            'real_metrics': real_metrics.__dict__,
            'challenge_analysis': challenge_analysis,
            'adaptation_recommendations': adaptation_recommendations,
            'validation_timestamp': time.time()
        }
    
    def _extract_metrics(self, results: Dict[str, Any], environment_type: str) -> TransferMetrics:
        """Extract standardized metrics from results"""
        # Handle different result formats
        if 'scenario_results' in results:
            # From model validation results
            return self._extract_from_validation_results(results)
        else:
            # From direct performance metrics
            return self._extract_from_performance_metrics(results)
    
    def _extract_from_validation_results(self, results: Dict[str, Any]) -> TransferMetrics:
        """Extract metrics from validation results"""
        scenario_results = results.get('scenario_results', {})
        
        # Aggregate metrics across scenarios
        success_rates = []
        precisions = []
        safety_violations = []
        completion_times = []
        
        for scenario_name, scenario_data in scenario_results.items():
            success_rates.append(scenario_data.get('success_rate', 0.0))
            precisions.append(scenario_data.get('average_precision', 0.0))
            safety_violations.append(scenario_data.get('safety_violation_rate', 1.0))
            
            if 'completion_time_avg' in scenario_data and scenario_data['completion_time_avg'] > 0:
                completion_times.append(scenario_data['completion_time_avg'])
        
        return TransferMetrics(
            success_rate=np.mean(success_rates) if success_rates else 0.0,
            navigation_precision=np.mean(precisions) if precisions else 0.0,
            safety_violations=np.mean(safety_violations) if safety_violations else 1.0,
            average_completion_time=np.mean(completion_times) if completion_times else 0.0,
            control_smoothness=0.8,  # Placeholder - would need specific measurement
            energy_efficiency=0.7,   # Placeholder - would need specific measurement
            robustness_score=results.get('validation_score', 0.0)
        )
    
    def _extract_from_performance_metrics(self, results: Dict[str, Any]) -> TransferMetrics:
        """Extract metrics from direct performance measurements"""
        return TransferMetrics(
            success_rate=results.get('success_rate', 0.0),
            navigation_precision=results.get('navigation_precision', 0.0),
            safety_violations=results.get('safety_violations', 1.0),
            average_completion_time=results.get('average_completion_time', 0.0),
            control_smoothness=results.get('control_smoothness', 0.0),
            energy_efficiency=results.get('energy_efficiency', 0.0),
            robustness_score=results.get('robustness_score', 0.0)
        )
    
    def _calculate_transfer_gaps(self, sim_metrics: TransferMetrics, real_metrics: TransferMetrics) -> Dict[str, float]:
        """Calculate performance gaps between simulation and real world"""
        gaps = {
            'success_rate_gap': abs(sim_metrics.success_rate - real_metrics.success_rate),
            'precision_gap': abs(sim_metrics.navigation_precision - real_metrics.navigation_precision),
            'safety_gap': abs(real_metrics.safety_violations - sim_metrics.safety_violations),  # Increase in violations
            'efficiency_gap': abs(sim_metrics.energy_efficiency - real_metrics.energy_efficiency),
            'control_gap': abs(sim_metrics.control_smoothness - real_metrics.control_smoothness),
            'time_gap': abs(real_metrics.average_completion_time - sim_metrics.average_completion_time) / max(sim_metrics.average_completion_time, 1.0),
            'robustness_gap': abs(sim_metrics.robustness_score - real_metrics.robustness_score)
        }
        
        # Calculate relative gaps (as percentages)
        relative_gaps = {}
        for gap_name, gap_value in gaps.items():
            if gap_name == 'safety_gap':
                # For safety, we care about absolute increase
                relative_gaps[gap_name] = gap_value
            else:
                # For other metrics, calculate relative to simulation performance
                sim_value = getattr(sim_metrics, gap_name.replace('_gap', ''))
                if sim_value > 0:
                    relative_gaps[gap_name + '_relative'] = gap_value / sim_value
                else:
                    relative_gaps[gap_name + '_relative'] = 1.0 if gap_value > 0 else 0.0
        
        gaps.update(relative_gaps)
        return gaps
    
    def _calculate_transfer_quality(self, transfer_gaps: Dict[str, float]) -> float:
        """Calculate overall transfer quality score"""
        quality_scores = {}
        
        # Success rate quality
        success_gap = transfer_gaps.get('success_rate_gap', 1.0)
        quality_scores['success_rate'] = max(0.0, 1.0 - (success_gap / 0.3))  # Normalize by 30% max acceptable gap
        
        # Safety quality (inverse of safety violations)
        safety_gap = transfer_gaps.get('safety_gap', 1.0)
        quality_scores['safety'] = max(0.0, 1.0 - (safety_gap / 0.1))  # Normalize by 10% max acceptable increase
        
        # Precision quality
        precision_gap = transfer_gaps.get('precision_gap', 1.0)
        quality_scores['precision'] = max(0.0, 1.0 - (precision_gap / 0.2))  # Normalize by 20% max acceptable gap
        
        # Efficiency quality
        efficiency_gap = transfer_gaps.get('efficiency_gap', 1.0)
        quality_scores['efficiency'] = max(0.0, 1.0 - (efficiency_gap / 0.3))  # Normalize by 30% max acceptable gap
        
        # Control quality
        control_gap = transfer_gaps.get('control_gap', 1.0)
        quality_scores['control_quality'] = max(0.0, 1.0 - (control_gap / 0.2))  # Normalize by 20% max acceptable gap
        
        # Calculate weighted overall quality
        overall_quality = sum(score * self.quality_weights[metric] 
                            for metric, score in quality_scores.items())
        
        return min(1.0, max(0.0, overall_quality))
    
    def _analyze_transfer_challenges(self, transfer_gaps: Dict[str, float], 
                                   sim_metrics: TransferMetrics, 
                                   real_metrics: TransferMetrics) -> Dict[str, Any]:
        """Analyze specific challenges in sim-to-real transfer"""
        challenges = {}
        
        # Identify primary challenge areas
        significant_gaps = {}
        for gap_name, threshold in self.transfer_gap_thresholds.items():
            gap_value = transfer_gaps.get(gap_name, 0.0)
            if gap_value > threshold:
                significant_gaps[gap_name] = {
                    'gap_value': gap_value,
                    'threshold': threshold,
                    'severity': min(1.0, gap_value / threshold)
                }
        
        challenges['significant_gaps'] = significant_gaps
        
        # Analyze performance degradation patterns
        degradation_analysis = {}
        
        # Success rate analysis
        if real_metrics.success_rate < sim_metrics.success_rate * 0.8:
            degradation_analysis['success_rate'] = {
                'type': 'significant_degradation',
                'sim_value': sim_metrics.success_rate,
                'real_value': real_metrics.success_rate,
                'likely_causes': ['sensor_noise', 'environmental_factors', 'actuator_delays']
            }
        
        # Safety analysis
        if real_metrics.safety_violations > sim_metrics.safety_violations + 0.1:
            degradation_analysis['safety'] = {
                'type': 'safety_degradation',
                'sim_value': sim_metrics.safety_violations,
                'real_value': real_metrics.safety_violations,
                'likely_causes': ['unexpected_obstacles', 'sensor_limitations', 'control_latency']
            }
        
        # Precision analysis
        if real_metrics.navigation_precision < sim_metrics.navigation_precision * 0.7:
            degradation_analysis['precision'] = {
                'type': 'precision_loss',
                'sim_value': sim_metrics.navigation_precision,
                'real_value': real_metrics.navigation_precision,
                'likely_causes': ['visual_conditions', 'calibration_drift', 'mechanical_tolerances']
            }
        
        challenges['degradation_analysis'] = degradation_analysis
        
        # Domain gap assessment
        domain_gaps = self._assess_domain_gaps(sim_metrics, real_metrics)
        challenges['domain_gaps'] = domain_gaps
        
        return challenges
    
    def _assess_domain_gaps(self, sim_metrics: TransferMetrics, real_metrics: TransferMetrics) -> Dict[str, str]:
        """Assess specific domain gaps between simulation and reality"""
        gaps = {}
        
        # Visual domain gap
        if abs(sim_metrics.navigation_precision - real_metrics.navigation_precision) > 0.15:
            gaps['visual_domain'] = 'significant_gap'
        elif abs(sim_metrics.navigation_precision - real_metrics.navigation_precision) > 0.08:
            gaps['visual_domain'] = 'moderate_gap'
        else:
            gaps['visual_domain'] = 'minimal_gap'
        
        # Dynamics domain gap
        if abs(sim_metrics.control_smoothness - real_metrics.control_smoothness) > 0.2:
            gaps['dynamics_domain'] = 'significant_gap'
        elif abs(sim_metrics.control_smoothness - real_metrics.control_smoothness) > 0.1:
            gaps['dynamics_domain'] = 'moderate_gap'
        else:
            gaps['dynamics_domain'] = 'minimal_gap'
        
        # Safety domain gap
        if real_metrics.safety_violations > sim_metrics.safety_violations + 0.1:
            gaps['safety_domain'] = 'significant_gap'
        elif real_metrics.safety_violations > sim_metrics.safety_violations + 0.05:
            gaps['safety_domain'] = 'moderate_gap'
        else:
            gaps['safety_domain'] = 'minimal_gap'
        
        return gaps
    
    def _generate_adaptation_recommendations(self, transfer_gaps: Dict[str, float], 
                                           challenge_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving sim-to-real transfer"""
        recommendations = []
        
        # Address significant gaps
        significant_gaps = challenge_analysis.get('significant_gaps', {})
        
        if 'success_rate_gap' in significant_gaps:
            recommendations.append("Consider domain randomization in simulation training to improve robustness")
            recommendations.append("Implement progressive difficulty curriculum with real-world noise models")
        
        if 'safety_gap' in significant_gaps:
            recommendations.append("Enhance safety margins and emergency protocols for real-world deployment")
            recommendations.append("Add conservative bias to control policies for safety-critical maneuvers")
        
        if 'precision_gap' in significant_gaps:
            recommendations.append("Improve visual system calibration and add sensor noise modeling")
            recommendations.append("Implement adaptive control to compensate for hardware variations")
        
        if 'control_gap' in significant_gaps:
            recommendations.append("Add actuator dynamics modeling and control system delays to simulation")
            recommendations.append("Implement online adaptation mechanisms for control system differences")
        
        # Address degradation patterns
        degradation_analysis = challenge_analysis.get('degradation_analysis', {})
        
        if 'success_rate' in degradation_analysis:
            recommendations.append("Extend training with environmental variations and sensor noise")
        
        if 'safety' in degradation_analysis:
            recommendations.append("Implement hierarchical safety controller with hardware-specific constraints")
        
        if 'precision' in degradation_analysis:
            recommendations.append("Perform in-situ calibration and visual system optimization")
        
        # Domain-specific recommendations
        domain_gaps = challenge_analysis.get('domain_gaps', {})
        
        if domain_gaps.get('visual_domain') == 'significant_gap':
            recommendations.append("Implement visual domain adaptation techniques")
        
        if domain_gaps.get('dynamics_domain') == 'significant_gap':
            recommendations.append("Add system identification and adaptive control components")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Transfer quality is good - proceed with standard deployment protocols")
        
        # Limit to most important recommendations
        return recommendations[:6]
    
    def _assess_transfer_success(self, transfer_quality: float, transfer_gaps: Dict[str, float]) -> bool:
        """Assess if sim-to-real transfer is successful"""
        # Check overall quality threshold
        if transfer_quality < 0.7:
            return False
        
        # Check critical safety gap
        safety_gap = transfer_gaps.get('safety_gap', 1.0)
        if safety_gap > self.transfer_gap_thresholds['safety_gap']:
            return False
        
        # Check success rate gap
        success_gap = transfer_gaps.get('success_rate_gap', 1.0)
        if success_gap > self.transfer_gap_thresholds['success_rate_gap']:
            return False
        
        return True
    
    def generate_transfer_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate human-readable transfer validation report"""
        report = []
        
        report.append("=== Sim-to-Real Transfer Validation Report ===\n")
        
        # Overall assessment
        transfer_quality = validation_results['transfer_quality']
        transfer_successful = validation_results['transfer_successful']
        
        report.append(f"Transfer Quality Score: {transfer_quality:.3f}")
        report.append(f"Transfer Status: {'PASSED' if transfer_successful else 'FAILED'}\n")
        
        # Performance gaps
        report.append("Performance Gaps:")
        transfer_gaps = validation_results['transfer_gaps']
        for gap_name, gap_value in transfer_gaps.items():
            if not gap_name.endswith('_relative'):
                threshold = self.transfer_gap_thresholds.get(gap_name, 0.0)
                status = "PASS" if gap_value <= threshold else "FAIL"
                report.append(f"  {gap_name}: {gap_value:.3f} (threshold: {threshold:.3f}) [{status}]")
        
        report.append("")
        
        # Recommendations
        recommendations = validation_results['adaptation_recommendations']
        if recommendations:
            report.append("Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"  {i}. {rec}")
        
        return "\n".join(report)
    
    def save_transfer_validation(self, validation_results: Dict[str, Any], output_path: str):
        """Save transfer validation results to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(output_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        # Save human-readable report
        report_path = output_file.with_suffix('.txt')
        with open(report_path, 'w') as f:
            f.write(self.generate_transfer_report(validation_results))
        
        logger.info(f"Transfer validation results saved to {output_file}")
        logger.info(f"Transfer validation report saved to {report_path}")
