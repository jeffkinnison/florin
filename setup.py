from setuptools import setup

setup(
    name='florin',
    version='0.1a1',
    description='Fast image segmentation without needing to learn a thing.',
    url='https://github.com/jeffkinnison/florin',
    author='Jeff Kinnison, Elia Shahbazi',
    author_email='jkinniso@nd.edu, ashahbaz@nd.edu',
    packages=['florin', 'florin.pipelines'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: MIT',
        'Topic :: Computer Vision :: Image Segmentation',
        'Topic :: Computer Vision :: Image Processing Pipeline',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    keywords='machine_learning hyperparameters distributed_computing',
    install_requires=[
        'h5py',
        'mpi4py',
        'numpy',
        'pathos',
        'scikit-image',
        'scipy',
    ]
)
