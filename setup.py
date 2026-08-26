"""Setup script for the eyecare-AI package.

See the README for usage instructions.
"""

from pathlib import Path

from setuptools import find_packages, setup

# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent.resolve()

# Read the long description from the README file.
try:
    LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")
except FileNotFoundError:
    LONG_DESCRIPTION = "EyeCare AI - AI-powered eye care application."

# Read the dependencies from requirements.txt.
# Each line is a dependency; comments and blank lines are skipped.
def _load_requirements() -> "list[str]":
    requirements = []
    requirements_file = HERE / "requirements.txt"
    if requirements_file.exists():
        for line in requirements_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-"):
                requirements.append(line)
    return requirements


INSTALL_REQUIRES = _load_requirements()

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------
setup(
    name="eyecare-ai",
    version="0.1.0",
    description="AI-powered eye care application.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="RANIT MADHU",
    author_email="ranitmadhu@example.com",
    url="https://github.com/RANIT-MADHU/eyecare-AI",
    license="MIT",
    classifiers=[
        # Full list: https://pypi.org/classifiers/
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            # Example: "eyecare=eyecare_ai.cli:main",
        ],
    },
)
