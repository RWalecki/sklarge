import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="sklarge",
    version="0.0.1",
    author="Robert Walecki",
    author_email="r.walecki14@imperial.ac.uk",
    description=(""),
    license = "BSD",
    keywords = "",
    url = "",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    data_files = [
        ('./sklarge/job_files/', ['./sklarge/job_files/execute.sh']),
        ('./sklarge/job_files/', ['./sklarge/job_files/run_condor.cmd']),
        ('./sklarge/job_files/', ['./sklarge/job_files/run_local.py']),
        ],
    packages=find_packages(),
)
