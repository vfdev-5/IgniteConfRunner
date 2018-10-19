import os
import io
import re
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


long_description = read("README.md")
version = find_version('ignite_conf_runner', '__init__.py')


setup(
    name="ignite_conf_runner",
    version=version,
    description=u"Ignite Configuration file Runner",
    long_description=long_description,
    author="vfdev-5",
    author_email="vfdev.5@gmail.com",
    url="https://github.com/vfdev-5/IgniteConfRunner",
    packages=find_packages(exclude=['tests', 'examples']),
    include_package_data=True,
    install_requires=[
        'numpy',
        'torch',
        'pytorch-ignite',
        'pandas',
        'mlflow',
        'pathlib2;python_version<"3"'
    ],
    license='MIT',
    test_suite="tests",
    tests_require=[
        'backports.tempfile;python_version<"3"',
        'pytest',
        'pytest-cov'        
    ],
    entry_points="""
        [console_scripts]
            ignite_run=ignite_conf_runner.runner:run_task
    """
)
