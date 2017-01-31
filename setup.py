from setuptools import setup, find_packages

setup(
    name='balsa',
    requires=[
        'pandas',
        'numpy',
        'numba',
        'numexpr',
        'astor'
    ],
    packages=find_packages()
)
