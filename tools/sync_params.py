import shutil
import textwrap
from pathlib import Path

def create_placeholder_md(path, yaml_filename):
    # The '\' at the start prevents an empty first line
    content = textwrap.dedent(f"""\
        # {path.stem.replace('_', ' ').title()}

        ```{{literalinclude}} {yaml_filename}
        :language: yaml
        :linenos:
        ```
    """)
    
    with open(path, "w", encoding="utf-8") as f: 
        f.write(content)

def sync_directory(source_dir, dest_dir, label):
    """Core worker function to sync params files and generate placeholder docs."""
    if not source_dir.exists():
        print(f"❌ {label} source not found: {source_dir}")
        return

    print(f"🚀 Syncing {label} from '{source_dir}'...")
    
    # Ensure destination exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy all YAML files
    for yaml_path in source_dir.glob("*.yaml"):
        dest_path = dest_dir / yaml_path.name
        
        # Only copy if changed (preserves modification times for Makefiles)
        shutil.copy2(yaml_path, dest_path)
        print(f"   📄 Copied {yaml_path.name}")
        
        # Auto-generate a placeholder .md if it doesn't exist
        md_path = dest_dir / f"{yaml_path.stem}.md"
        if not md_path.exists():
            print(f"   ✨ Creating placeholder MD for {md_path.name}")
            create_placeholder_md(md_path, yaml_path.name, label)

def main():
    # Configuration dictionary mapping the Label to (Source Path, Destination Path)
    sync_targets = {
        "Examples": (Path("src/ptyrad/starter/params/examples"), Path("docs/examples")),
        "Walkthroughs": (Path("src/ptyrad/starter/params/walkthrough"), Path("docs/walkthrough")),
        "Templates": (Path("src/ptyrad/starter/params/templates"), Path("docs/templates")),
    }

    # Loop through each target and run the worker function
    for label, (src, dest) in sync_targets.items():
        sync_directory(src, dest, label)
        print()  # Add a blank line between categories for cleaner terminal output

if __name__ == "__main__":
    main()