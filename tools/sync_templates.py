import shutil
from pathlib import Path
import textwrap

SOURCE_TEMPLATES = Path("src/ptyrad/starter/params/templates/")
DEST_DOCS_TEMPLATES = Path("docs/templates")

def create_placeholder_md(path, yaml_filename):
    # The '\' at the start prevents an empty first line
    content = textwrap.dedent(f"""\
        # {path.stem.replace('_', ' ').title()}
        
        Description of this template goes here.

        ```{{literalinclude}} {yaml_filename}
        :language: yaml
        :linenos:
        ```
    """)
    
    with open(path, "w", encoding="utf-8") as f: 
        f.write(content)

def main():
    if not SOURCE_TEMPLATES.exists():
        print(f"❌ Examples source not found: {SOURCE_TEMPLATES}")
        return

    print(f"🚀 Syncing Exampless from '{SOURCE_TEMPLATES}'...")
    
    # Ensure destination exists
    DEST_DOCS_TEMPLATES.mkdir(parents=True, exist_ok=True)

    # Copy all YAML files
    for yaml_path in SOURCE_TEMPLATES.glob("*.yaml"):
        dest_path = DEST_DOCS_TEMPLATES / yaml_path.name
        
        # Only copy if changed (preserves modification times for Makefiles)
        # or just simple copy2 is fine for Sphinx
        shutil.copy2(yaml_path, dest_path)
        print(f"   📄 Copied {yaml_path.name}")
        
        # OPTIONAL: Auto-generate a placeholder .md if it doesn't exist?
        # This helps you get started quickly.
        md_path = DEST_DOCS_TEMPLATES / f"{yaml_path.stem}.md"
        if not md_path.exists():
            print(f"   ✨ Creating placeholder MD for {md_path.name}")
            create_placeholder_md(md_path, yaml_path.name)
        
if __name__ == "__main__":
    main()