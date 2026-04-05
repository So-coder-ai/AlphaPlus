"""
AlphaPulse - AI-Powered Crypto Futures Trading Bot

Setup configuration for pip installation and distribution.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="alphapulse",
    version="1.0.0",
    author="AlphaPulse Team",
    author_email="contact@alphapulse.ai",
    description="AI-powered cryptocurrency futures trading bot with machine learning signals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alphapulse",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "sentiment": [
            "requests>=2.31.0",
            "transformers>=4.37.0",
            "torch>=2.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "alphapulse=cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "cryptocurrency",
        "trading",
        "bot",
        "machine learning",
        "futures",
        "binance",
        "technical analysis",
        "backtesting",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/alphapulse/issues",
        "Source": "https://github.com/yourusername/alphapulse",
        "Documentation": "https://github.com/yourusername/alphapulse/blob/main/README.md",
    },
)
