import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="IMNN",
    version="0.2a1",
    author="Tom Charnock",
    author_email="charnock@iap.fr",
    description="Using neural networks to extract sufficient statistics from \
        data by maximising the Fisher information",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tomcharnock/information_maximiser.git",
    packages=["IMNN", "IMNN.ABC", "IMNN.utils"],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3",
    install_requires=[
          "tensorflow>=2.0.0",
          "tqdm>=4.31.0",
          "numpy>=1.16.0"
      ],
)
