"""
Setup configuration for ML Trading Bot

This setup.py file allows the ML Trading Bot to be installed as a Python package.
Install with: pip install -e .
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    """Read README.md file for package description (from parent directory)."""
    readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "ML Trading Bot - Automated trading using machine learning models"

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements.txt file for dependencies (from parent directory)."""
    requirements_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
    requirements = []
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments, empty lines, and optional dependencies
                if line and not line.startswith('#') and not line.startswith('-'):
                    # Remove inline comments
                    if '#' in line:
                        line = line[:line.index('#')].strip()
                    if line:
                        requirements.append(line)
    return requirements

setup(
    # Basic package information
    name="ml-trading-bot",
    version="1.0.0",
    author="ML Trading Bot Project",
    author_email="your-email@example.com",  # Update with actual email
    description="Automated trading bot using machine learning for stock prediction and sentiment analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ML-trading-bot",  # Update with actual URL
    license="GPL-3.0",
    
    # Package discovery (setup.py is now in src/)
    packages=find_packages(where="."),
    package_dir={"": "."},
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "ml-trading-bot=main:main",
            "tsr-model-train=tsr_model.train:main",
            "sentiment-tokenize=sentiment_model.tokenize_pipeline:main",
        ],
    },
    
    # Package classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    # Additional metadata
    keywords="trading, machine-learning, finance, sentiment-analysis, time-series",
    project_urls={
        "Bug Reports": "https://github.com/your-username/ML-trading-bot/issues",
        "Source": "https://github.com/your-username/ML-trading-bot",
        "Documentation": "https://github.com/your-username/ML-trading-bot/wiki",
    },
    
    # Include package data
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yaml", "*.yml"],
    },
    
    # Development dependencies (optional)
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0", 
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.15.0",
            "ipywidgets>=8.0.0",
        ],
        "web": [
            "streamlit>=1.22.0",
            "flask>=2.2.0",
        ],
    },
    
    # Zip safe
    zip_safe=False,
)