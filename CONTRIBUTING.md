# Contributing to AlphaPulse

Thank you for your interest in contributing to AlphaPulse! This document provides guidelines for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Binance Testnet API keys (for testing)
- Git

### Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/alphapulse.git
   cd alphapulse
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## 📝 Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and small

### Testing

- Run tests before submitting:
  ```bash
  pytest tests/ -v --cov=.
  ```
- Add tests for new features
- Ensure test coverage remains above 80%

### Documentation

- Update README.md for user-facing changes
- Add inline comments for complex logic
- Update docstrings for new functions

## 🔄 Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and test thoroughly
3. Commit your changes with descriptive messages
4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Create a Pull Request

### PR Requirements

- Clear description of changes
- Tests pass
- Code follows project style
- Documentation is updated
- No breaking changes without discussion

## 🐛 Bug Reports

When reporting bugs, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages (if any)

## 💡 Feature Requests

Feature requests are welcome! Please:

- Open an issue with the "enhancement" label
- Describe the use case
- Explain why it would be valuable
- Consider implementation suggestions

## 📜 Code of Conduct

Be respectful and inclusive:
- Use welcoming language
- Be constructive in feedback
- Respect different viewpoints
- Focus on what is best for the community

## 🏷️ Issue Labels

- `bug`: Bug reports
- `enhancement`: Feature requests
- `documentation`: Doc improvements
- `good first issue`: Good for newcomers
- `help wanted`: Need assistance

## 📞 Questions?

Feel free to:
- Open an issue for questions
- Start a discussion in the repository
- Check existing issues and discussions

Thank you for contributing! 🎉
