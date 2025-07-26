"""
Test script to verify imports for the Financial Intelligence module.
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to Python path: {project_root}")

try:
    # Try importing the financial_intelligence package
    import financial_intelligence
    print("\nSuccessfully imported financial_intelligence package!")
    print(f"Version: {financial_intelligence.__version__}")
    
    # List available functions
    print("\nAvailable functions:")
    for func in financial_intelligence.__all__:
        print(f"- {func}")
    
    # Try creating an instance of a class from a submodule
    from financial_intelligence.portfolio_opt.optimizer import PortfolioOptimizer
    print("\nSuccessfully imported PortfolioOptimizer class!")
    
    # Try importing a function from the facade
    from financial_intelligence import get_portfolio_performance_metrics
    print("Successfully imported get_portfolio_performance_metrics function!")
    
    print("\nAll imports successful! The financial_intelligence package is properly set up.")
    
except ImportError as e:
    print(f"\nError importing module: {e}")
    print("\nCurrent Python path:")
    for path in sys.path:
        print(f"- {path}")
    
    print("\nCurrent working directory:")
    print(os.getcwd())
