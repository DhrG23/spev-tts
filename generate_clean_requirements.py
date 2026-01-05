#!/usr/bin/env python3
"""
Generate clean requirements.txt from conda environment
This removes local file paths and keeps only PyPI package names with versions
"""

import re
import sys

def clean_requirements(input_file='requirements_conda.txt', output_file='requirements.txt'):
    """
    Convert conda-style requirements to pip-compatible requirements
    """
    
    # Core packages we actually need for SPEV TTS
    core_packages = {
        'torch', 'torchaudio', 'librosa', 'soundfile', 'numpy', 'scipy',
        'pandas', 'textgrid', 'requests', 'tqdm', 'pyyaml', 'scikit-learn',
        'joblib', 'numba', 'cffi', 'packaging', 'praatio', 'soxr',
        'audioread', 'noisereduce', 'matplotlib', 'pillow'
    }
    
    clean_reqs = []
    
    try:
        with open(input_file, 'r', encoding='utf-16') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {input_file} not found!")
        print("Usage: python generate_clean_requirements.py [input_file] [output_file]")
        return
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Extract package name and version
        # Pattern: package_name==version or package_name @ file://...
        if ' @ file://' in line:
            # Extract package name before @ symbol
            package_name = line.split(' @ ')[0].strip()
            
            # Get version if available in the path
            version_match = re.search(r'_(\d+\.\d+\.\d+)', line)
            if version_match:
                version = version_match.group(1)
                clean_line = f"{package_name}>={version}"
            else:
                clean_line = package_name
        elif '==' in line:
            clean_line = line
        else:
            continue
        
        # Normalize package name (convert underscores and hyphens)
        package_base = clean_line.split('>=')[0].split('==')[0].lower().replace('_', '-')
        
        # Only include core packages
        if any(core in package_base for core in core_packages):
            clean_reqs.append(clean_line)
    
    # Add packages that might be missing
    required_packages = [
        'torch>=2.0.0',
        'torchaudio>=2.0.0',
        'librosa>=0.10.0',
        'soundfile>=0.12.1',
        'numpy>=1.24.0,<2.0.0',
        'scipy>=1.10.0',
        'pandas>=2.0.0',
        'textgrid>=1.5.0',
        'requests>=2.31.0',
        'tqdm>=4.65.0',
    ]
    
    # Check if required packages are in clean_reqs
    for req in required_packages:
        pkg_name = req.split('>=')[0].split('==')[0]
        if not any(pkg_name in r for r in clean_reqs):
            clean_reqs.append(req)
    
    # Sort and write
    clean_reqs = sorted(set(clean_reqs))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# SPEV TTS - Requirements\n")
        f.write("# Auto-generated clean requirements for pip installation\n\n")
        f.write("# Deep Learning Framework\n")
        
        for req in clean_reqs:
            if 'torch' in req.lower():
                f.write(f"{req}\n")
        
        f.write("\n# Audio Processing\n")
        for req in clean_reqs:
            if any(x in req.lower() for x in ['librosa', 'soundfile', 'audioread', 'soxr']):
                f.write(f"{req}\n")
        
        f.write("\n# Scientific Computing\n")
        for req in clean_reqs:
            if any(x in req.lower() for x in ['numpy', 'scipy', 'numba']):
                f.write(f"{req}\n")
        
        f.write("\n# Data Processing\n")
        for req in clean_reqs:
            if 'pandas' in req.lower():
                f.write(f"{req}\n")
        
        f.write("\n# Text Processing & Phoneme Support\n")
        for req in clean_reqs:
            if any(x in req.lower() for x in ['textgrid', 'praatio']):
                f.write(f"{req}\n")
        
        f.write("\n# Machine Learning\n")
        for req in clean_reqs:
            if any(x in req.lower() for x in ['scikit-learn', 'joblib']):
                f.write(f"{req}\n")
        
        f.write("\n# Utilities\n")
        remaining = [r for r in clean_reqs if not any(x in r.lower() for x in 
                    ['torch', 'librosa', 'soundfile', 'audioread', 'soxr', 
                     'numpy', 'scipy', 'numba', 'pandas', 'textgrid', 'praatio',
                     'scikit-learn', 'joblib'])]
        for req in remaining:
            f.write(f"{req}\n")
    
    print(f"✓ Clean requirements written to {output_file}")
    print(f"✓ Found {len(clean_reqs)} packages")
    print("\nTo install:")
    print(f"  pip install -r {output_file}")

if __name__ == '__main__':
    if len(sys.argv) > 2:
        clean_requirements(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        clean_requirements(sys.argv[1])
    else:
        clean_requirements()