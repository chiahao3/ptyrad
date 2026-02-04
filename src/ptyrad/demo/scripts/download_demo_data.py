import os
import sys
import time
import zipfile
import urllib.request
import ssl
from pathlib import Path

# --- CONFIGURATION ---
# Format: { "Folder_Name_On_Disk": "Direct_Download_URL" }
DATASETS = {
    # For Box files, set to shared with link, get the link, put /shared/static before file ID, and add '.zip' to the end
    "PSO": "https://cornell.box.com/shared/static/b8u6e7mqfwmcjx4s3u4ohgrhpgb9tcsl.zip",
    "tBL_WSe2": "https://cornell.box.com/shared/static/7rpcj4syh0l724k8sk07l24rjqi5zyc6.zip",
}

def show_progress(block_num, block_size, total_size):
    """Callback for urllib to display a text-based progress bar."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        bar_length = 30
        filled_length = int(bar_length * percent / 100)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f'\r⬇️  Downloading: [{bar}] {percent:.1f}% ')
        sys.stdout.flush()

def validate_zip_header(file_path):
    """Checks if file is a valid ZIP or an HTML redirect page."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
        if header == b'PK\x03\x04':
            return True, "Valid ZIP header"
        
        with open(file_path, 'r', errors='ignore') as f:
            start_text = f.read(200)
            if "<!DOCTYPE html>" in start_text or "<html" in start_text:
                return False, "Detected HTML webpage instead of ZIP file."
        return False, f"Unknown file header: {header}"
    except Exception as e:
        return False, f"Validation error: {e}"

def process_dataset(name, url, data_root):
    """Downloads and extracts a single dataset."""
    target_dir = data_root / name
    zip_path = data_root / f"temp_{name}.zip"
    
    print(f"\n🔹 Processing dataset: '{name}'")
    
    # 1. Check existence
    if target_dir.exists():
        print(f"   ✅ Already exists at: {target_dir}")
        return True

    print(f"   📡 Connecting to: {url}")
    
    # SSL Context
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # 2. Download
    try:
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, zip_path, reporthook=show_progress)
        print("") # Clear progress line
    except Exception as e:
        print(f"\n   ❌ Download failed: {e}")
        if zip_path.exists(): os.remove(zip_path)
        return False

    # 3. Validate
    is_valid, reason = validate_zip_header(zip_path)
    if not is_valid:
        print(f"   ❌ Validation failed: {reason}")
        os.remove(zip_path)
        return False

    # 4. Extract
    print("   📦 Extracting...", end=" ")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_root)
        print("Done!")
    except Exception as e:
        print(f"\n   ❌ Extraction error: {e}")
        return False
    finally:
        if zip_path.exists(): os.remove(zip_path)
    
    print(f"   ✨ Ready at: {target_dir}")
    return True

def main():
    # Setup Paths: /scripts/ -> /data/
    current_dir = Path(__file__).resolve().parent
    data_root = current_dir.parent / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    success_count = 0
    
    print(f"🚀 Starting download for {len(DATASETS)} dataset(s)...")
    
    for name, url in DATASETS.items():
        if process_dataset(name, url, data_root):
            success_count += 1
            
    total_time = time.time() - start_time
    print("-" * 50)
    if success_count == len(DATASETS):
        print(f"✅ All datasets ready! (Total time: {total_time:.1f}s)")
    else:
        print(f"⚠️  Completed {success_count}/{len(DATASETS)} datasets. Check errors above.")

if __name__ == "__main__":
    main()