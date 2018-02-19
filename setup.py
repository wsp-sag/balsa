from setuptools import setup, find_packages

setup(
    name='balsa',
    version='0.5.6',
    packages=find_packages(),
    requires=[
        'pandas',
        'numpy',
        'astor',
        'numba',
        'numexpr',
        'six'
    ]
)
