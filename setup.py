from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="inavvi_v11",
    version="0.1.0",
    author="INAVVI Team",
    author_email="info@inavvi.com",
    description="Autonomous Trading System with AI/ML capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Slusheeking/INAVVI_v11",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "inavvi-start=src.cli.start:main",
            "inavvi-stop=src.cli.stop:main",
            "inavvi-status=src.cli.status:main",
            "inavvi-backtest=src.cli.backtest:main",
        ],
    },
)