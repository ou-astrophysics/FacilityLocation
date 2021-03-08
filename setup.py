import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(packages=["FacilityLocation"], install_requires=["numba", "numpy"])
