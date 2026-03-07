#!/usr/bin/env python3
"""
EasyGPU Code Formatter

Formats all C++ source files (.h, .hpp, .cpp) using clang-format
with the project's .clang-format configuration.

Usage:
    python format-code.py              # Format all files in place
    python format-code.py --dry-run    # Show which files would be formatted
    python format-code.py --check      # Check if files are formatted (CI mode)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def get_project_root() -> Path:
    """Get the project root directory (where this script is located)."""
    return Path(__file__).parent.absolute()


def find_source_files(project_root: Path) -> List[Path]:
    """Find all C++ source files in the project."""
    source_dirs = ["include", "source", "tests", "examples"]
    extensions = [".h", ".hpp", ".cpp"]
    
    files = []
    for dir_name in source_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            for ext in extensions:
                files.extend(dir_path.rglob(f"*{ext}"))
    
    # Sort for consistent ordering
    return sorted(files)


def check_clang_format(project_root: Path) -> Tuple[bool, str]:
    """Check if clang-format is available and config exists."""
    # Check config file
    config_file = project_root / ".clang-format"
    if not config_file.exists():
        return False, ".clang-format not found in project root"
    
    # Check clang-format binary
    try:
        result = subprocess.run(
            ["clang-format", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout.strip()
    except FileNotFoundError:
        return False, "clang-format not found in PATH. Please install LLVM/clang-format."
    except subprocess.CalledProcessError as e:
        return False, f"clang-format error: {e}"


def format_file(file_path: Path, config_path: Path, check_mode: bool = False) -> Tuple[bool, bool]:
    """
    Format a single file.
    
    Returns:
        Tuple of (success, needs_formatting)
        - success: True if operation succeeded
        - needs_formatting: True if file needs formatting (only valid in check_mode)
    """
    try:
        if check_mode:
            # Check mode: read original and compare with formatted output
            with open(file_path, 'r', encoding='utf-8') as f:
                original = f.read()
            
            result = subprocess.run(
                ["clang-format", f"-style=file:{config_path}", str(file_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            formatted = result.stdout
            needs_formatting = original != formatted
            return True, needs_formatting
        else:
            # Format mode: format in place
            result = subprocess.run(
                ["clang-format", "-i", f"-style=file:{config_path}", str(file_path)],
                capture_output=True,
                text=True,
                check=True
            )
            return True, False
            
    except subprocess.CalledProcessError as e:
        return False, False
    except Exception as e:
        return False, False


def main():
    parser = argparse.ArgumentParser(
        description="Format EasyGPU C++ source files using clang-format"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be formatted without actually formatting"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if files are formatted correctly. Returns exit code 1 if any files need formatting (useful for CI)"
    )
    args = parser.parse_args()
    
    project_root = get_project_root()
    
    print("=" * 50)
    print("EasyGPU Code Formatter")
    print(f"Project Root: {project_root}")
    print("=" * 50)
    print()
    
    # Check prerequisites
    ok, message = check_clang_format(project_root)
    if not ok:
        print(f"Error: {message}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Config file: .clang-format")
    print(f"Using: {message}")
    print()
    
    # Find source files
    files = find_source_files(project_root)
    total = len(files)
    
    if total == 0:
        print("No files found!")
        sys.exit(0)
    
    print(f"Found {total} files to process:")
    print()
    for f in files:
        relative = f.relative_to(project_root)
        print(f"  - {relative}")
    print()
    
    # Dry run mode
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        print(f"Would format {total} files")
        sys.exit(0)
    
    # Process files
    config_path = project_root / ".clang-format"
    success_count = 0
    fail_count = 0
    needs_formatting = []
    
    for i, file_path in enumerate(files, 1):
        relative = file_path.relative_to(project_root)
        progress = f"[{i}/{total}]"
        
        print(f"{progress} Processing: {relative} ... ", end="", flush=True)
        
        success, needs_format = format_file(file_path, config_path, args.check)
        
        if success:
            if args.check:
                if needs_format:
                    needs_formatting.append(str(relative))
                    print("[NEEDS FORMAT]")
                else:
                    print("[OK]")
                    success_count += 1
            else:
                print("[DONE]")
                success_count += 1
        else:
            print("[FAILED]")
            fail_count += 1
    
    print()
    print("=" * 50)
    
    if args.check:
        if needs_formatting:
            print(f"CHECK FAILED: {len(needs_formatting)} files need formatting")
            print()
            print("Files requiring formatting:")
            for f in needs_formatting:
                print(f"  - {f}")
            sys.exit(1)
        else:
            print(f"CHECK PASSED: All {success_count} files are properly formatted")
            sys.exit(0)
    else:
        print("Formatting Complete!")
        print(f"  Success: {success_count}")
        if fail_count > 0:
            print(f"  Failed:  {fail_count}")
    
    print("=" * 50)
    
    sys.exit(1 if fail_count > 0 else 0)


if __name__ == "__main__":
    main()
