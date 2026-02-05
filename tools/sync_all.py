import sys
from pathlib import Path

# Ensure we can import siblings by adding tools/ to path explicitly
# (Useful if you ever run this from weird locations)
sys.path.append(str(Path(__file__).parent))

import sync_notebooks
import sync_walkthrough
import sync_examples

def sync_all():
    print("🔄 Starting full synchronization...")
    
    print("\n[1/3] Syncing Notebooks...")
    sync_notebooks.main()
    
    print("\n[2/3] Syncing Walkthroughs...")
    sync_walkthrough.main()
    
    print("\n[3/3] Syncing Walkthroughs...")
    sync_examples.main()
    
    print("\n✨ All systems go!")

if __name__ == "__main__":
    sync_all()