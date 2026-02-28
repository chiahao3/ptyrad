import sys
from pathlib import Path

# Ensure we can import siblings by adding tools/ to path explicitly
# (Useful if you ever run this from weird locations)
sys.path.append(str(Path(__file__).parent))

import sync_notebooks
import sync_walkthrough
import sync_examples
import sync_templates

def sync_all():
    print("🔄 Starting full synchronization...")
    
    print("\n[1] Syncing Notebooks...")
    sync_notebooks.main()
    
    print("\n[2] Syncing Walkthroughs...")
    sync_walkthrough.main()
    
    print("\n[3] Syncing Examples...")
    sync_examples.main()
    
    print("\n[4] Syncing Templates...")
    sync_templates.main()
    
    print("\n✨ All systems go!")

if __name__ == "__main__":
    sync_all()