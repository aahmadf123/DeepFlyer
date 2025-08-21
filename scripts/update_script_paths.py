#!/usr/bin/env python3
"""
Update script paths after reorganization
Updates hardcoded paths in training scripts to match new structure
"""

import os
import re
from pathlib import Path

def update_file_paths(file_path, path_mappings):
    """Update file paths in a given file"""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    for old_path, new_path in path_mappings.items():
        content = content.replace(old_path, new_path)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated paths in {file_path}")
    else:
        print(f"No path updates needed in {file_path}")

def main():
    """Update all script paths"""
    
    # Common path mappings after reorganization
    path_mappings = {
        # Model paths
        '"weights/best.pt"': '"trained_models/yolo/best.pt"',
        "'weights/best.pt'": "'trained_models/yolo/best.pt'",
        "weights/best.pt": "trained_models/yolo/best.pt",
        
        # YOLO model paths  
        "yolo11n.pt": "datasets/yolo_models/yolo11n.pt",
        "yolo11m.pt": "datasets/yolo_models/yolo11m.pt",
        
        # Output directories
        '"models"': '"trained_models/p3o"',
        "'models'": "'trained_models/p3o'",
        "Path(\"models\")": "Path(\"trained_models/p3o\")",
        "Path('models')": "Path('trained_models/p3o')",
        
        # Log directories  
        '"logs"': '"experiments/logs"',
        "'logs'": "'experiments/logs'",
        "Path(\"logs\")": "Path(\"experiments/logs\")",
        "Path('logs')": "Path('experiments/logs')",
    }
    
    # Files to update
    files_to_update = [
        "scripts/train_p3o.py",
        "scripts/hyperopt_runner.py", 
        "scripts/validate_environment.py",
        "scripts/validate_hardware.py",
        "config/training_config.yaml",
        "rl_agent/px4_training_node.py",
        "rl_agent/direct_control_agent.py"
    ]
    
    print("=== Updating Script Paths ===")
    
    for file_path in files_to_update:
        update_file_paths(file_path, path_mappings)
    
    print("\n=== Path Updates Complete ===")
    print("Verify scripts work with new directory structure")

if __name__ == "__main__":
    main()
