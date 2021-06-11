from setuptools import setup, find_packages

import versioneer

setup(
    name='wsp-balsa',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Python tools for travel demand forecasting applications and analyses',
    url='https://github.com/wsp-sag/balsa',
    author='WSP',
    maintainer='Brian Cheung',
    maintainer_email='brian.cheung@wsp.com',
    classifiers=[
        'License :: OSI Approved :: MIT License'
    ],
    packages=find_packages(),
    install_requires=[
        'pandas>=0.21',
        'numpy>=1.15',
        'numba>=0.35',
        'numexpr>=2.6'
    ],
    python_requires='>=3.5',
    extras_require={
        'plotting': 'matplotlib>=3.0'
    }
)
