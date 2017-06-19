from setuptools import setup, find_packages
from os import path


def process_line(line):
    if '#' in line:
        first_comment_char = line.index('#')
        line = line[:first_comment_char]
    return line.strip()

fp = path.join(path.dirname(__file__), "requirements.txt")
with open(fp) as reader:
    installation_requirements = [process_line(line) for line in reader if not line.startswith('#')]

setup(
    name='balsa',
    version='0.3',
    requires=installation_requirements,
    packages=find_packages()
)
