from setuptools import find_packages, setup

setup(
    name='ml_project',
    version='1.0.0',
    description='ml project',
    packages=find_packages(exclude=['tests*', ]),
    package_data={},
)
