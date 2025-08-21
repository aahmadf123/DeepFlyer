#!/usr/bin/env python3
"""
DeepFlyer Codebase Reorganization Script
Moves files to proper directory structure for better organization
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def create_directory_structure():
    """Create the new organized directory structure"""
    directories = [
        "datasets/yolo_models",
        "datasets/training_data", 
        "trained_models/yolo",
        "trained_models/p3o", 
        "trained_models/checkpoints",
        "experiments/runs",
        "experiments/mlflow",
        "docs/integration",
        "docs/architecture"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def move_model_files():
    """Move scattered model files to organized locations"""
    moves = [
        # YOLO pre-trained models to datasets
        ("yolo11n.pt", "datasets/yolo_models/yolo11n.pt"),
        ("yolo11m.pt", "datasets/yolo_models/yolo11m.pt"),
        
        # Training datasets to datasets directory  
        ("Drone Gates.v1-version1.yolov11.zip", "datasets/training_data/drone_gates_v1.zip"),
        ("Hoops data.v2i.yolov11.zip", "datasets/training_data/hoops_data_v2.zip"),
        ("Racing-Gate.v4i.yolov11.zip", "datasets/training_data/racing_gate_v4.zip"),
        
        # Best model weights to models directory
        ("weights/best.pt", "trained_models/yolo/best.pt")
    ]
    
    for src, dst in moves:
        if os.path.exists(src):
            print(f"Moving {src} -> {dst}")
            shutil.move(src, dst)
        else:
            print(f"Warning: {src} not found")

def move_training_outputs():
    """Move training runs to organized experiments directory"""
    runs_dir = Path("runs")
    if runs_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_location = f"experiments/runs/training_archive_{timestamp}"
        print(f"Moving training runs: {runs_dir} -> {new_location}")
        shutil.move(str(runs_dir), new_location)
        
    # Move MLflow if it exists in runs
    mlflow_path = Path("experiments/runs/training_archive_" + timestamp + "/mlflow")
    if mlflow_path.exists():
        shutil.move(str(mlflow_path), "experiments/mlflow")
        print("Moved MLflow data to experiments/mlflow")

def reorganize_documentation():
    """Consolidate documentation into organized structure"""
    doc_moves = [
        # Integration guides to docs/integration
        ("INTEGRATION_GUIDE.md", "docs/integration/main_integration_guide.md"),
        ("UMA_INTEGRATION_GUIDE.md", "docs/integration/uma_integration_guide.md"), 
        ("api/JAY_INTEGRATION_GUIDE.md", "docs/integration/jay_integration_guide.md"),
        ("TEAM_OVERVIEW.md", "docs/integration/team_overview.md"),
        
        # Architecture docs to docs/architecture  
        ("docs/DEEPFLYER_CONCEPT.md", "docs/architecture/deepflyer_concept.md"),
        ("docs/APPROACH_EVOLUTION.md", "docs/architecture/approach_evolution.md"),
        ("docs/PX4_RL_IMPLEMENTATION.md", "docs/architecture/px4_rl_implementation.md"),
        
        # Keep these in main docs for easy access
        ("docs/QUICKSTART.md", "docs/quickstart.md"),
        ("docs/MISSING_COMPONENTS_GUIDE.md", "docs/missing_components_guide.md"),
        ("docs/YOLO11_INTEGRATION_GUIDE.md", "docs/yolo11_integration_guide.md"),
        ("docs/Plans.md", "docs/development_plans.md")
    ]
    
    for src, dst in doc_moves:
        if os.path.exists(src):
            print(f"Moving documentation: {src} -> {dst}")
            # Ensure destination directory exists
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)

def update_path_configurations():
    """Update configuration files with new paths"""
    # Update training config if it references old paths
    config_file = "config/training_config.yaml"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Update common path references
        content = content.replace("weights/best.pt", "trained_models/yolo/best.pt")
        content = content.replace("yolo11", "datasets/yolo_models/yolo11")
        
        with open(config_file, 'w') as f:
            f.write(content)
        print("Updated training_config.yaml paths")

def cleanup_empty_directories():
    """Remove empty directories after reorganization"""
    empty_dirs = ["weights", "docs/images"]
    
    for directory in empty_dirs:
        if os.path.exists(directory) and not os.listdir(directory):
            os.rmdir(directory)
            print(f"Removed empty directory: {directory}")

def create_gitignore_updates():
    """Add new directories to .gitignore if needed"""
    gitignore_additions = [
        "\n# Organized experiment outputs",
        "experiments/runs/",
        "experiments/mlflow/",
        "\n# Model artifacts (keep structure, ignore large files)",
        "models/*/[!.gitkeep]*", 
        "datasets/training_data/*.zip",
        "datasets/yolo_models/*.pt"
    ]
    
    gitignore_path = ".gitignore"
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'a') as f:
            f.write('\n'.join(gitignore_additions))
        print("Updated .gitignore with new directory patterns")

def create_readme_updates():
    """Create README files for new directories"""
    readme_content = {
        "datasets/README.md": """# Datasets Directory

## Structure
- `yolo_models/`: Pre-trained YOLO model weights
- `training_data/`: Course detection training datasets

## Usage
YOLO models are automatically downloaded during training.
Training datasets should be placed in training_data/ directory.
""",
        
        "models/README.md": """# Models Directory  

## Structure
- `yolo/`: Trained YOLO detection models
- `p3o/`: Trained P3O reinforcement learning models
- `checkpoints/`: Training checkpoint files

## Usage
Best models are saved here after training completion.
Use these paths in your training configurations.
""",
        
        "experiments/README.md": """# Experiments Directory

## Structure  
- `runs/`: Archived training run outputs
- `mlflow/`: MLflow experiment tracking data

## Usage
All training outputs and experiment tracking data is organized here.
Use MLflow UI to view training metrics and comparisons.
"""
    }
    
    for file_path, content in readme_content.items():
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Created {file_path}")

def main():
    """Execute the complete reorganization"""
    print("=== DeepFlyer Codebase Reorganization ===")
    print("This script will reorganize files for better structure")
    
    response = input("Continue with reorganization? [y/N]: ")
    if response.lower() != 'y':
        print("Reorganization cancelled")
        return
    
    print("\n1. Creating directory structure...")
    create_directory_structure()
    
    print("\n2. Moving model files...")
    move_model_files()
    
    print("\n3. Moving training outputs...")
    move_training_outputs()
    
    print("\n4. Reorganizing documentation...")
    reorganize_documentation()
    
    print("\n5. Updating configurations...")
    update_path_configurations()
    
    print("\n6. Cleaning up empty directories...")
    cleanup_empty_directories()
    
    print("\n7. Updating .gitignore...")
    create_gitignore_updates()
    
    print("\n8. Creating README files...")
    create_readme_updates()
    
    print("\n=== Reorganization Complete ===")
    print("New structure:")
    print("- datasets/: YOLO models and training data")
    print("- models/: Trained model outputs") 
    print("- experiments/: Training runs and MLflow data")
    print("- docs/: Organized documentation")
    print("\nUpdate your scripts to use the new file paths!")

if __name__ == "__main__":
    main()
