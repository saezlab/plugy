from setuptools import setup

setup(
    name='plugy',
    version='0.7.0',
    packages=['plugy', 'plugy.data', 'plugy.test'],
    url='https://github.com/saezlab/plugy',
    license='GPLv3',
    author='Dénes Türei, Nicolas Peschke, Olga Ivanova',
    author_email='turei.denes@gmail.com',
    description='Processing plug microfluidics data',
    install_requires=[
        'seaborn',
        'matplotlib',
        'statsmodels',
        'scipy',
        'pandas',
        'numpy',
        'scikit-image',
        'tqdm',
    ],
)
