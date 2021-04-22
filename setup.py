import setuptools
# import io

# with io.open("README.rst", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="imnn",
    version="0.3dev",
    author="Tom Charnock",
    author_email="charnock@iap.fr",
    description="Using neural networks to extract sufficient statistics from \
        data by maximising the Fisher information",
    # long_description=long_description,
    # long_description_content_type="text/reStructuredText",
    url="https://bitbucket.org/tomcharnock/IMNN.git",
    packages=["imnn", "imnn.imnn", "imnn.utils", "test", "examples", "docs"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "jax>=0.2.10",
        "tensorflow>=2.1.0",
        "tqdm>=4.48.2",
        "tensorflow_probability",
        "numpy",
        "scipy",
        "matplotlib"],
)
