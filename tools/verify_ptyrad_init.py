import sys
import subprocess
import shutil
from pathlib import Path
import ptyrad.starter

def get_relative_files(directory):
    """Returns a set of relative file paths, ignoring caches and pyc files."""
    if not directory.exists():
        return set()
    return {
        f.relative_to(directory).as_posix()
        for f in directory.rglob('*')
        if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc'
    }

def main():
    if len(sys.argv) < 2:
        print("❌ ERROR: Output directory argument missing.")
        print("Usage: python tools/verify_ptyrad_init.py <output_directory>")
        sys.exit(1)

    out_dir_path = sys.argv[1]
    out_dir = Path(out_dir_path).resolve()

    # --- EXECUTE CLI COMMAND ---
    # Clean up any previous failed runs to ensure a pristine test
    if out_dir.exists():
        shutil.rmtree(out_dir)

    print(f"🚀 Running 'ptyrad init {out_dir_path}' via subprocess...")
    try:
        subprocess.run(
            ["ptyrad", "init", out_dir_path],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ SUCCESS: 'ptyrad init' executed without crashing.\n")
    except subprocess.CalledProcessError as e:
        print("❌ ERROR: 'ptyrad init' command failed.")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ ERROR: 'ptyrad' command not found. Is the package installed?")
        sys.exit(1)
    # ---------------------------

    # Define the three directories to compare
    src_dir = Path('src/ptyrad/starter').resolve()
    pkg_dir = Path(ptyrad.starter.__file__).parent.resolve()

    # Gather file sets
    src_files = get_relative_files(src_dir)
    pkg_files = get_relative_files(pkg_dir)
    out_files = get_relative_files(out_dir)

    # Print generated files
    print(f'\n--- Generated contents of {out_dir.name}/ ---')
    data_file_count = 0

    # f is already a relative path string here!
    file_list = list(out_files)
    file_list.sort()
    for f in file_list:
        print(f'  - {f}')
        if not f.endswith('.py'):
            data_file_count += 1
    print('----------------------------------------\n')

    print('\n--- File Count Summary ---')
    print(f'Source directory:  {len(src_files)} files')
    print(f'Installed package: {len(pkg_files)} files')
    print(f'Generated output:  {len(out_files)} files\n')
    print("Note: `ptyrad init` will not copy __init__.py inside soucre and installed package.")
    print('--------------------------\n')
    
    errors = 0

    # Check 1: Was src/ptyrad/starter correctly included in the build?
    missing_in_pkg = src_files - pkg_files
    if missing_in_pkg:
        print('❌ ERROR: Files missing from the installed package:')
        for f in missing_in_pkg:
            print(f'  - {f}')
        errors += 1
    else:
        print('✅ Check 1 Passed: Build includes all starter files.')

    # Check 2: Did ptyrad init copy all files (excluding __init__.py)?
    expected_out_files = {f for f in src_files if not f.endswith('__init__.py')}
    missing_in_out = expected_out_files - out_files

    if missing_in_out:
        print('❌ ERROR: Files missing from the generated output directory:')
        for f in missing_in_out:
            print(f'  - {f}')
        errors += 1
    else:
        print('✅ Check 2 Passed: init command correctly copied all packaged files.')

    if errors > 0:
        sys.exit(1)
    else:
        # Optional: Clean up the output directory if everything passed perfectly
        shutil.rmtree(out_dir)
        print("\n🎉 Verification Passed. Temporary output files cleaned up.")

if __name__ == "__main__":
    main()

# import sys
# from pathlib import Path
# import ptyrad.starter

# def get_relative_files(directory):
#     """Returns a set of relative file paths, ignoring caches and pyc files."""
#     if not directory.exists():
#         return set()
#     return {
#         f.relative_to(directory).as_posix()
#         for f in directory.rglob('*')
#         if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc'
#     }

# def main():
#     if len(sys.argv) < 2:
#         print("❌ ERROR: Output directory argument missing.")
#         print("Usage: python tools/verify_ptyrad_init.py <output_directory>")
#         sys.exit(1)

#     out_dir_path = sys.argv[1]

#     # Define the three directories to compare
#     src_dir = Path('src/ptyrad/starter').resolve()
#     pkg_dir = Path(ptyrad.starter.__file__).parent.resolve()
#     out_dir = Path(out_dir_path).resolve()

#     # Gather file sets
#     src_files = get_relative_files(src_dir)
#     pkg_files = get_relative_files(pkg_dir)
#     out_files = get_relative_files(out_dir)

#     # Print generated files
#     print(f'\n--- Generated contents of {out_dir.name}/ ---')
#     data_file_count = 0

#     # f is already a relative path string here!
#     file_list = list(out_files)
#     file_list.sort()
#     for f in file_list:
#         print(f'  - {f}')
#         if not f.endswith('.py'):
#             data_file_count += 1
#     print('----------------------------------------\n')

#     print('\n--- File Count Summary ---')
#     print(f'Source directory:  {len(src_files)} files')
#     print(f'Installed package: {len(pkg_files)} files')
#     print(f'Generated output:  {len(out_files)} files\n')
#     print("Note: `ptyrad init` will not copy __init__.py inside soucre and installed package.")
#     print('--------------------------\n')
    
#     errors = 0

#     # Check 1: Was src/ptyrad/starter correctly included in the build?
#     missing_in_pkg = src_files - pkg_files
#     if missing_in_pkg:
#         print('❌ ERROR: Files missing from the installed package:')
#         for f in missing_in_pkg:
#             print(f'  - {f}')
#         errors += 1
#     else:
#         print('✅ Check 1 Passed: Build includes all starter files.')

#     # Check 2: Did ptyrad init copy all files (excluding __init__.py)?
#     expected_out_files = {f for f in src_files if not f.endswith('__init__.py')}
#     missing_in_out = expected_out_files - out_files

#     if missing_in_out:
#         print('❌ ERROR: Files missing from the generated output directory:')
#         for f in missing_in_out:
#             print(f'  - {f}')
#         errors += 1
#     else:
#         print('✅ Check 2 Passed: init command correctly copied all user files.')

#     if errors > 0:
#         sys.exit(1)

# if __name__ == "__main__":
#     main()