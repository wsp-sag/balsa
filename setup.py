from setuptools import setup, find_packages

setup(
    name='balsa',
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
