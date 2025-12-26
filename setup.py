"""Setup script for Ocean SST Super-Resolution package."""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='ocean-sst-superres',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='CNN-based super-resolution for ocean sea surface temperature data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/ocean-sst-superres',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'flake8>=6.0.0',
            'black>=23.0.0',
            'isort>=5.12.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'sst-download=data.download:main',
            'sst-coarsen=data.coarsen:main',
            'sst-patches=data.patches:main',
            'sst-train=train:main',
            'sst-inference=inference:main',
        ],
    },
)
