from setuptools import setup, find_packages

setup(
    name='balsa',
    version='0.2',
    requires=[
        'pandas',
        'numpy',
        'numba',
        'numexpr',
        'astor',
        'six'
    ],
    packages=find_packages()
)
