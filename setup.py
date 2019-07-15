from setuptools import setup, find_packages

setup(
    name='takos',
    version='0.0.1',
    url='https://github.com/Taekyoon/takos-alpha',
    license='MIT',
    author='Taekyoon Choi',
    author_email='tgchoi03@gmail.om',
    description='Trainable Korean spacing library alpha version',
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
)