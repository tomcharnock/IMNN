import setuptools
import io

with io.open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="IMNN",
    version="0.3.1",
    author="Tom Charnock",
    author_email="charnock@iap.fr",
    description="Using neural networks to extract sufficient statistics from \
        data by maximising the Fisher information",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://bitbucket.org/tomcharnock/imnn.git",
    packages=["imnn", "imnn.imnn", "imnn.lfi", "imnn.utils"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "jax>=0.2.13",
        "jaxlib>=0.1.62",
        "tensorflow",
        "tqdm>=4.48.2",
        "tensorflow-probability[jax]",
        "numpy",
        "scipy",
        "matplotlib"],
    include_package_data=True
)
