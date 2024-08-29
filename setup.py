from os.path import join, dirname, abspath
from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    long_description = f.read()

setup(
    name='logbatcher',
    version='0.1.1',
    description='A tool for batch parsing of log files',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['logbatcher'],
    install_requires=requirements,
    # extras_require={"default": requirements},
)
    
