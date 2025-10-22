"""Setup configuration for LowNoCompute-AI-Baseline package."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='lownocompute-ai-baseline',
    version='1.0.0',
    author='Sunghun Kwag',
    author_email='your.email@example.com',  # Update with actual email
    description='A minimal, modular AI baseline framework for meta-learning under strict low-compute constraints',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/sunghunkwag/LowNoCompute-AI-Baseline',
    project_urls={
        'Bug Tracker': 'https://github.com/sunghunkwag/LowNoCompute-AI-Baseline/issues',
        'Documentation': 'https://github.com/sunghunkwag/LowNoCompute-AI-Baseline/blob/main/README.md',
        'Source Code': 'https://github.com/sunghunkwag/LowNoCompute-AI-Baseline',
    },
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*', 'docs', 'docs.*']),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'black>=21.0',
            'flake8>=3.9.0',
        ],
    },
    keywords='meta-learning, state-space-models, experience-buffer, low-compute, AI, machine-learning, MAML',
    include_package_data=True,
    package_data={
        'lownocompute_ai_baseline': ['configs/*.yaml'],
    },
    zip_safe=False,
)

