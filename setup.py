from os.path import join, dirname, abspath
from setuptools import setup

with open(join(dirname(abspath(__file__)), 'requirements.txt')) as f:
    requirements = f.read().splitlines()

with open(join(dirname(abspath(__file__)), 'README.md')) as f:
    long_description = f.read()

setup(
    name='logbatcher',
    version='0.1.0',
    description='A tool for batch parsing of log files',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['logbatcher'],
    install_requires=[],
    extras_require={"default": requirements},
)
    
