import sys
from pathlib import Path

# Ensure we can import siblings by adding tools/ to path explicitly
# (Useful if you ever run this from weird locations)
sys.path.append(str(Path(__file__).parent))

import sync_notebooks
import sync_params

def sync_all():
    print("🔄 Starting full synchronization...")
    
    print("\n[1] Syncing Notebooks...")
    sync_notebooks.main()
    
    print("\n[2] Syncing Params (Examples, Walkthrough, Templates)...")
    sync_params.main()
    
    print("\n✨ All systems go!")

if __name__ == "__main__":
    sync_all()