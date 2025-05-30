"""
This is a legacy setup.py file that's kept for compatibility.
Modern installation should use pyproject.toml with pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# This setup.py is maintained for compatibility
# The main configuration is in pyproject.toml
setup(
    name="eagleapi",
    version="0.1.0",
    author="Iyke David",
    author_email="davidiyke04@gmail.com",
    description="A modern Python web framework built on FastAPI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/niceiyke/eagle",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn[standard]>=0.15.0",
        "sqlalchemy>=1.4.0",
        "alembic>=1.7.0",
        "pydantic>=1.8.0",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "python-multipart>=0.0.5",
        "typer>=0.4.0",
        "rich>=10.0.0",
        "httpx>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.15.0",
            "pytest-cov>=2.0.0",
            "black>=21.0",
            "isort>=5.0.0",
            "mypy>=0.910",
        ],
    },
)
