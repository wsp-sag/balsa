from setuptools import setup, find_packages

setup(
    name='wsp-balsa',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'pandas>=0.21, <0.24',
        'numpy>=1.15',
        'numba>=0.35',
        'numexpr>=2.6',
        'six>=1.10',
        'matplotlib>=3.0'
    ]
)
