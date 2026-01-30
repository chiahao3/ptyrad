"""
CLI utility functions for setting up fresh project folder, and exporting templates / examples params files

"""

import shutil
import sys
from pathlib import Path
from importlib import resources

def create_starter_project(project_name: str = "ptyrad", force: bool = False):
    """
    Copies the internal 'demo' folder to a new directory named project_name.
    """
    dest = Path(project_name).resolve()
    
    # 1. Safety Check: Don't overwrite unless forced
    if dest.exists():
        if not force:
            print(f"Error: Directory '{dest.name}' already exists.")
            print(f"  Use 'ptyrad init {dest.name} --force' to overwrite it.")
            sys.exit(1)
        else:
            print(f"  Overwriting existing directory: {dest}")
            shutil.rmtree(dest)

    # 2. Locate the internal demo folder
    # We look for the package 'ptyrad.demo'
    try:
        source_ref = resources.files("ptyrad.demo")
    except (ImportError, TypeError):
         # Fallback for localized dev installs or broken packages
         print("Error: Could not locate internal demo files.")
         sys.exit(1)

    # 3. Copy the Tree
    # copytree creates the destination directory for us.
    try:
        with resources.as_file(source_ref) as src_path:
            shutil.copytree(
                src_path, 
                dest, 
                ignore=shutil.ignore_patterns("__*", "*.pyc") # Skip __init__.py and cache
            )
            
        print(f"Created starter project at: {dest}")
        print( "   Structure:")
        print(f"   ├── {dest.name}/data/     (Place your data here)")
        print(f"   ├── {dest.name}/output/   (Check results here)")
        print(f"   ├── {dest.name}/params/   (Template params files)")
        print(f"   └── {dest.name}/scripts/  (Starter scripts)")
        
    except Exception as e:
        print(f"Failed to create project: {e}")
        sys.exit(1)
        
def _export_resource(resource_subpath: str, dest_name: str, dest_parent: str = ".", force: bool = False, description: str = "files"):
    """
    Internal helper to copy a specific subfolder from ptyrad.demo to a local destination.
    
    Args:
        resource_subpath: Path inside ptyrad.demo (e.g., "params" or "params/templates")
        dest_name: Name of the folder to create locally (e.g., "params" or "templates")
        dest_parent: Local directory where dest_name will be created
        force: Whether to overwrite existing folders
        description: Text description for print messages
    """
    # 1. Setup paths
    target_path = Path(dest_parent).resolve() / dest_name
    
    # 2. Check for conflicts
    if target_path.exists():
        if not force:
            print(f"Error: Directory '{target_path.name}' already exists at {target_path.parent}")
            print("   Use --force to overwrite: -f")
            sys.exit(1)
        else:
            print(f"Overwriting existing {dest_name} folder: {target_path}")
            try:
                shutil.rmtree(target_path, ignore_errors=False)
            except OSError as e:
                print(f"Error removing existing folder: {e}")
                sys.exit(1)

    # 3. Copy the resource
    try:
        # Access the root 'ptyrad.demo' package
        demo_root = resources.files("ptyrad.demo")
        
        # Drill down to the specific subfolder (e.g. demo/params or demo/params/templates)
        # We split by '/' to handle nested paths like "params/templates" safely
        source_path = demo_root
        for part in resource_subpath.split("/"):
            source_path = source_path.joinpath(part)
        
        if not source_path.is_dir():
            print(f"Error: Could not locate '{resource_subpath}' folder inside package.")
            sys.exit(1)

        # Copy files
        with resources.as_file(source_path) as src_path:
            shutil.copytree(src_path, target_path)

        print(f"{description.capitalize()} exported to: {target_path}")
        
        # 1. Search for both .yaml AND .yml
        # rglob returns a generator, so we chain them or use a set comprehension
        yaml_files = list(target_path.rglob('*.yaml'))
        yml_files = list(target_path.rglob('*.yml'))
        all_files = sorted(yaml_files + yml_files)

        # 2. Convert to relative paths (e.g. "templates/minimal.yaml")
        # This makes it obvious where files are located
        file_list = [str(f.relative_to(target_path)) for f in all_files]
        
        if file_list:
            print("Files:")
            for file in file_list:
                print(f"  {file}")
        else:
            print("   (No YAML files found in export)")

    except Exception as e:
        print(f"Failed to export {description}: {e}")
        sys.exit(1)

def export_params(dest_dir: str = ".", force: bool = False):
    """
    Copies the entire 'demo/params/' folder (including templates/ and examples/) locally.
    """
    _export_resource(
        resource_subpath="params",
        dest_name="params",
        dest_parent=dest_dir,
        force=force,
        description="Parameter files"
    )

def export_templates(dest_dir: str = ".", force: bool = False):
    """
    Copies only the 'demo/params/templates/' folder locally.
    """
    _export_resource(
        resource_subpath="params/templates",
        dest_name="templates",
        dest_parent=dest_dir,
        force=force,
        description="Clean templates"
    )

def export_examples(dest_dir: str = ".", force: bool = False):
    """
    Copies only the 'demo/params/examples/' folder locally.
    """
    _export_resource(
        resource_subpath="params/examples",
        dest_name="examples",
        dest_parent=dest_dir,
        force=force,
        description="Explicit examples"
    )