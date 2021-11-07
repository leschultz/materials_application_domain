import setuptools

# Package information
name = 'mad'
version = '0.0.1'
description = 'Application domain of machine learning in materials science.'
url = 'https://github.com/leschultz/application_domain.git'
author = 'Lane E. Schultz'
author_email = 'laneenriqueschultz@gmail.com'
python_requires = '>=3.6'
classifiers = ['Programming Language :: Python :: 3',
               'License :: OSI Approved :: MIT License',
               'Operating System :: OS Independent',
               ]
packages = setuptools.find_packages()
install_requires = [
                    'matplotlib',
                    'scipy',
                    'scikit-learn',
                    'pandas',
                    'numpy',
                    'seaborn',
                    'pathos',
                    'tqdm',
                    'pytest',
                    'openpyxl',
                    'statsmodels's
                    ]
long_description = open('README.md').read()

# Passing variables to setup
setuptools.setup(
                 name=name,
                 version=version,
                 description=description,
                 url=url,
                 author=author,
                 author_email=author_email,
                 packages=packages,
                 python_requires=python_requires,
                 classifiers=classifiers,
                 install_requires=install_requires,
                 long_description=long_description,
                 )
