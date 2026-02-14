#!/usr/bin/env python3
import os
import subprocess
import glob
from pathlib import Path

def validate_all_params():
    """
    Finds and validates all YAML parameter files in the project.
    """
    # Define the search root relative to this script
    project_root = Path(__file__).parent.parent
    params_dir = project_root / "src/ptyrad/starter/params"
    
    # Define patterns for .yaml and .yml files
    # This covers subfolders like examples/, templates/, and walkthrough/
    yaml_files = glob.glob(str(params_dir / "**" / "*.yaml"), recursive=True)
    yaml_files += glob.glob(str(params_dir / "**" / "*.yml"), recursive=True)

    if not yaml_files:
        print(f"No YAML files found in {params_dir}")
        return

    print(f"Found {len(yaml_files)} parameter files. Starting validation...\n")

    success_count = 0
    for file_path in yaml_files:
        relative_path = os.path.relpath(file_path, project_root)
        
        # Construct the CLI command
        # We use 'ptyrad' directly assuming it's installed or in the path
        command = ["ptyrad", "validate-params", file_path]
        
        try:
            # Run the command and capture output
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(f"✅ VALID: {relative_path}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ FAILED: {relative_path}")
            print(f"   Error: {e.stderr.strip()}")

    print(f"\nSummary: {success_count}/{len(yaml_files)} files passed validation.")

if __name__ == "__main__":
    validate_all_params()