from setuptools import setup, find_packages

setup(
    name='wsp-balsa',
    author='wsp',
    maintatiner='Brian Cheung',
    maintainer_email='brian.cheung@wsp.com',
    version='1.1',
    packages=find_packages(),
    python_requires='>=3.5',
    install_requires=[
        'pandas>=0.21',
        'numpy>=1.15',
        'numba>=0.35',
        'numexpr>=2.6',
        'six>=1.10'
    ],
    extras_require={
        'plotting': 'matplotlib>=3.0'
    }
)
