import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cvdatasetutils_dmoranj",
    version="0.0.4",
    author="Daniel Morán Jiménez",
    author_email="dmoranj@gmail.com",
    description="Utils to download and manage a selection of Computer Vision datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dmoranj/cvdatasetutils",
    packages=setuptools.find_packages(),
    install_requires=[
        'networkx',
        'matplotlib',
        'mltrainingutils',
        'nltk',
        'multiset',
        'sklearn',
        'spacy',
        'skimage',
        'torch',
	    'Pillow',
        'torchvision'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)
