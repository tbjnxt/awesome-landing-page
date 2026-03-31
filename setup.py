"""Setup for offline_claw — installable via pip install -e ."""
from setuptools import setup, find_packages

setup(
    name="offline_claw",
    version="0.1.0",
    description="Offline Claude Code harness powered by Ollama",
    packages=find_packages(),
    package_data={"offline_claw": ["reference_data/*.json"]},
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "offline-claw=offline_claw.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
