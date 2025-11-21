#!/usr/bin/env python3
"""
Verify MolRAG installation and file integrity

Checks:
- All required files exist
- Files are not empty
- Modules can be imported
- Dependencies are installed
"""

import sys
from pathlib import Path
import subprocess

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def check_mark(condition):
    """Return check mark or X based on condition"""
    return f"{GREEN}✓{RESET}" if condition else f"{RED}✗{RESET}"


def check_files():
    """Check if all required files exist and are not empty"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Checking File Structure{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    required_files = {
        # Core
        'src/__init__.py': 100,
        'src/molrag.py': 300,

        # Data
        'src/data/__init__.py': 50,
        'src/data/fingerprints.py': 300,
        'src/data/preprocessor.py': 200,
        'src/data/gnn_embeddings.py': 300,
        'src/data/kg_loader.py': 400,
        'src/data/models.py': 100,

        # Retrieval
        'src/retrieval/__init__.py': 50,
        'src/retrieval/vector_retrieval.py': 150,
        'src/retrieval/graph_retrieval.py': 250,
        'src/retrieval/gnn_retrieval.py': 150,
        'src/retrieval/reranker.py': 250,
        'src/retrieval/triple_retriever.py': 150,

        # Reasoning
        'src/reasoning/__init__.py': 50,
        'src/reasoning/agents.py': 500,
        'src/reasoning/cot_strategies.py': 400,
        'src/reasoning/orchestrator.py': 200,

        # Evaluation
        'src/evaluation/__init__.py': 50,
        'src/evaluation/metrics.py': 400,

        # Utils
        'src/utils/__init__.py': 50,
        'src/utils/config.py': 100,
        'src/utils/database.py': 350,
        'src/utils/logger.py': 50,

        # Config
        'config/models.yaml': 200,
        'config/knowledge_graphs.yaml': 200,

        # UI
        'app.py': 600,

        # Docs
        'README.md': 300,
        'requirements.txt': 50,
    }

    all_good = True
    total_lines = 0

    for filepath, min_lines in required_files.items():
        path = Path(filepath)
        exists = path.exists()

        if exists:
            # Count lines
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                total_lines += lines

                if lines >= min_lines:
                    status = f"{GREEN}✓{RESET}"
                    size_info = f"({lines} lines)"
                else:
                    status = f"{YELLOW}⚠{RESET}"
                    size_info = f"({lines} lines, expected {min_lines}+)"
                    all_good = False
            except Exception as e:
                status = f"{RED}✗{RESET}"
                size_info = f"(read error: {e})"
                all_good = False
        else:
            status = f"{RED}✗{RESET}"
            size_info = "(missing)"
            all_good = False

        print(f"{status} {filepath:50s} {size_info}")

    print(f"\n{BLUE}Total lines of code:{RESET} {total_lines}")

    return all_good


def check_imports():
    """Check if core modules can be imported"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Checking Module Imports{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    modules = [
        'src.data.fingerprints',
        'src.data.preprocessor',
        'src.data.models',
        'src.utils.config',
        'src.utils.logger',
    ]

    all_good = True

    for module in modules:
        try:
            __import__(module)
            print(f"{GREEN}✓{RESET} {module}")
        except Exception as e:
            print(f"{RED}✗{RESET} {module} - Error: {e}")
            all_good = False

    return all_good


def check_dependencies():
    """Check if required dependencies are installed"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Checking Dependencies{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    critical_packages = [
        'rdkit',
        'numpy',
        'pandas',
        'pydantic',
        'loguru',
        'gradio',
    ]

    optional_packages = [
        'torch',
        'torch_geometric',
        'neo4j',
        'qdrant_client',
        'redis',
        'openai',
        'anthropic',
    ]

    all_critical = True

    print(f"{YELLOW}Critical Dependencies:{RESET}")
    for package in critical_packages:
        try:
            __import__(package)
            print(f"{GREEN}✓{RESET} {package}")
        except ImportError:
            print(f"{RED}✗{RESET} {package} - NOT INSTALLED")
            all_critical = False

    print(f"\n{YELLOW}Optional Dependencies:{RESET}")
    for package in optional_packages:
        try:
            __import__(package.replace('_', '.') if '_' in package else package)
            print(f"{GREEN}✓{RESET} {package}")
        except ImportError:
            print(f"{YELLOW}⚠{RESET} {package} - not installed (optional)")

    return all_critical


def check_git_branch():
    """Check current git branch"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Git Branch Information{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    try:
        result = subprocess.run(
            ['git', 'branch', '--show-current'],
            capture_output=True,
            text=True,
            check=True
        )
        branch = result.stdout.strip()

        expected_branch = "claude/review-project-codebase-01XHMakVV7QgEdagpdU78VWB"

        if branch == expected_branch:
            print(f"{GREEN}✓{RESET} Current branch: {branch}")
            print(f"{GREEN}✓{RESET} Correct branch!")
            return True
        else:
            print(f"{YELLOW}⚠{RESET} Current branch: {branch}")
            print(f"{YELLOW}⚠{RESET} Expected branch: {expected_branch}")
            print(f"\n{YELLOW}Run:{RESET} git checkout {expected_branch}")
            return False

    except Exception as e:
        print(f"{RED}✗{RESET} Could not determine git branch: {e}")
        return False


def main():
    """Run all verification checks"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}MolRAG Installation Verification{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    # Check branch
    branch_ok = check_git_branch()

    # Check files
    files_ok = check_files()

    # Check imports
    imports_ok = check_imports()

    # Check dependencies
    deps_ok = check_dependencies()

    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Verification Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    print(f"Git Branch: {check_mark(branch_ok)}")
    print(f"File Structure: {check_mark(files_ok)}")
    print(f"Module Imports: {check_mark(imports_ok)}")
    print(f"Dependencies: {check_mark(deps_ok)}")

    if all([branch_ok, files_ok, imports_ok, deps_ok]):
        print(f"\n{GREEN}{'='*60}{RESET}")
        print(f"{GREEN}✓ All checks passed! MolRAG is ready to use.{RESET}")
        print(f"{GREEN}{'='*60}{RESET}")
        print(f"\n{BLUE}Next steps:{RESET}")
        print(f"1. Launch UI: {YELLOW}python app.py{RESET}")
        print(f"2. Setup databases: {YELLOW}python scripts/setup_databases.py{RESET}")
        print(f"3. See examples: {YELLOW}python examples/basic_usage.py{RESET}")
        return 0
    else:
        print(f"\n{RED}{'='*60}{RESET}")
        print(f"{RED}✗ Some checks failed. Please review above.{RESET}")
        print(f"{RED}{'='*60}{RESET}")
        print(f"\n{YELLOW}Troubleshooting:{RESET}")

        if not branch_ok:
            print(f"- Checkout correct branch: git checkout claude/review-project-codebase-01XHMakVV7QgEdagpdU78VWB")

        if not files_ok:
            print(f"- See PULL_INSTRUCTIONS.md for how to pull files")

        if not deps_ok:
            print(f"- Install dependencies: pip install -r requirements.txt")

        return 1


if __name__ == "__main__":
    sys.exit(main())
