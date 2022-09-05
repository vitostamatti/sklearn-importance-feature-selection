from setuptools import setup, find_packages


setup(
    name='importance_feature_selector',
    version='0.1.0',
    description='feature selection based on feature importance',
    # long_description=long_description,
    license="MIT",
    author='Vito Stamatti',
    package_dir={'':'.'},
    packages=find_packages(where='.'),
    install_requires=[
        'scikit-learn', 
    ],
),