from setuptools import setup, find_packages

setup(
    name='balsa',
    version='0.6.1',
    packages=find_packages(),
    install_requires=[
        'pandas>=0.21',
        'numpy>=1.16',
        'astor>=0.5',
        'numba>=0.35',
        'numexpr>=2.6',
        'six>=1.10'
    ]
)
