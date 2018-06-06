from setuptools import setup

import numpy as np

with open("README.rst", "r") as f:
    README_TEXT = f.read()

setup(name="nnkcde",
      version="0.1",
      license="MIT",
      description="Fits nearest neighbor kernel conditional density estimates",
      long_description = README_TEXT,
      author           = "Taylor Pospisil",
      author_email     = "tpospisi@andrew.cmu.edu",
      maintainer       = "tpospisi@andrew.cmu.edu",
      url="https://github.com/tpospisi/nnkcde/python",
      classifiers = ["License :: OSI Approved :: MIT License",
                     "Topic :: Scientific/Engineering :: Artificial Intelligence",
                     "Programming Language :: Python :: 2.7",
                     "Programming Language :: Python :: 3.6"],
      keywords = ["conditional density estimation", "nearest neigbhor"],
      package_dir={"": "src"},
      packages=["nnkcde"],
      python_requires=">=2.7",
      install_requires=["numpy", "scipy", "sklearn"],
      setup_requires=["pytest-runner"],
      tests_require=["pytest"],
      zip_safe=False,
      include_package_data=True,
)
