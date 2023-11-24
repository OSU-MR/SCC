from setuptools import setup, find_packages

setup(
    name='Surface Coil Intensity Correction',
    version='0.5',
    packages=['helper_functions'],
    # metadata to display on PyPI
    author='Xuan Lei',
    author_email='lei.337@osu.edu',
    description='brightness correction map generator for siemens rawdata(twix)',
    long_description=open('README.md').read(),
    url='https://github.com/OSU-MR/SCC',
    classifiers=[
        'License :: all-rights-reserved'
    ],
    python_requires='>=3.8.10',
)
