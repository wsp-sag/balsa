from setuptools import setup, find_packages

setup(
    name='balsa',
    requires=[
        'pandas',
        'numpy',
        'numba',
        'numexpr'
    ],
    packages=find_packages()
)
