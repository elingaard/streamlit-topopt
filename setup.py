from pathlib import Path
from setuptools import setup

name = "st_topopt"
version = "0.0.0"
install_requires = Path("requirements.txt").read_text().splitlines()
dev_requires = Path("requirements-dev.txt").read_text().splitlines()
description = "Perform topology optimization with a web-application (Streamlit)"
maintainer_email = "m.elingaard@gmail.com"

package_dir = {"": "src"}
packages = ["st_topopt"]

extras_require = {"dev": install_requires + dev_requires}

entry_points = {}

setup(
    name=name,
    version=version,
    description=description,
    install_requires=install_requires,
    extras_require=extras_require,
    packages=packages,
    package_dir=package_dir,
    entry_points=entry_points,
)
