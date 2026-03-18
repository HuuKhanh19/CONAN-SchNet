from setuptools import setup, find_packages

setup(
    name="conan-schnet",
    version="0.1.0",
    description="CONAN-SchNet: Molecular Property Prediction with SchNet + EGGROLL + GP",
    author="Huu Khanh",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "schnetpack>=2.0.0",
        "rdkit>=2023.03.1",
        "ase>=3.22.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
    ],
)
