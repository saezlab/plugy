from setuptools import setup

setup(
    name='plugy',
    version='v0.4.4',
    packages=['plugy', 'plugy.data', 'plugy.test'],
    url='https://github.com/saezlab/plugy',
    license='GPLv3',
    author='Dénes Türei & Nicolas Peschke',
    author_email='turei.denes@gmail.com',
    description='A package to handle plug microfluidics data',
    install_requires=['seaborn',
                      'matplotlib',
                      'statsmodels',
                      'scipy',
                      'pandas',
                      'numpy',
                      'scikit-image',
                      ]
)
