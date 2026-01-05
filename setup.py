"""
SPEV TTS - Setup Configuration
Installation: pip install -e .
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements from requirements.txt
def read_requirements(filename='requirements.txt'):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Core requirements (if requirements.txt doesn't exist)
CORE_REQUIREMENTS = [
    'torch>=2.0.0',
    'torchaudio>=2.0.0',
    'librosa>=0.10.0',
    'soundfile>=0.12.0',
    'numpy>=1.24.0',
    'textgrid>=1.5.0',
    'cmudict>=1.0.0',
    'pandas>=2.0.0',
    'requests>=2.31.0',
]

# Optional dependencies for different use cases
EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
        'black>=23.0.0',
        'flake8>=6.0.0',
        'isort>=5.12.0',
        'mypy>=1.0.0',
    ],
    'docs': [
        'sphinx>=6.0.0',
        'sphinx-rtd-theme>=1.2.0',
        'sphinxcontrib-napoleon>=0.7',
    ],
    'training': [
        'tensorboard>=2.12.0',
        'wandb>=0.15.0',
        'tqdm>=4.65.0',
    ],
    'alignment': [
        'montreal-forced-aligner>=2.2.0',
        'praatio>=6.0.0',
    ],
}

# Add 'all' option that includes everything
EXTRAS_REQUIRE['all'] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

setup(
    name='spev-tts',
    version='1.0.0',
    description='SPEV: Advanced Text-to-Speech with Emotional Voice Control',
    long_description=read_file('README.md') if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    author='SPEV Development Team',
    author_email='your.email@example.com',
    url='https://github.com/dhrg23/spev-tts',
    license='MIT',
    
    # Package configuration
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'examples']),
    py_modules=['spev_tts', 'spev_advanced', 'dataset_loader'],
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Dependencies
    install_requires=read_requirements() if os.path.exists('requirements.txt') else CORE_REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'spev-train=spev_tts:main',
            'spev-infer=spev_tts:inference_mode',
            'spev-advanced-train=spev_advanced:main',
            'spev-advanced-infer=spev_advanced:inference_mode',
            'spev-download=download_datasets:main',
        ],
    },
    
    # Package data to include
    package_data={
        '': [
            '*.md',
            '*.txt',
            '*.sh',
            '*.ps1',
            'LICENSE',
        ],
    },
    
    # Include non-Python files
    include_package_data=True,
    
    # PyPI classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    
    # Keywords for PyPI
    keywords=[
        'text-to-speech',
        'tts',
        'speech-synthesis',
        'fastspeech2',
        'neural-tts',
        'voice-synthesis',
        'deep-learning',
        'pytorch',
        'emotional-speech',
        'voice-control',
    ],
    
    # Project URLs
    project_urls={
        'Documentation': 'https://github.com/yourusername/spev-tts#readme',
        'Source': 'https://github.com/yourusername/spev-tts',
        'Bug Reports': 'https://github.com/yourusername/spev-tts/issues',
    },
    
    # Zip safe
    zip_safe=False,
)