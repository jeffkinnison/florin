import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='florin',
    version='0.0.1',
    description='Fast image segmentation without needing to learn a thing.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/jeffkinnison/florin',
    author='Jeff Kinnison, Elia Shahbazi',
    author_email='jkinniso@nd.edu, ashahbaz@nd.edu',
    packages=['florin', 'florin.pipelines'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: System :: Distributed Computing',
    ],
    keywords='machine_learning hyperparameters distributed_computing',
    install_requires=[
        'h5py',
        'mpi4py>=3.0.0',
        'numpy',
        'pathos',
        'scikit-image',
        'scipy',
    ]
)
