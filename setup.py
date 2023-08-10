from setuptools import setup, find_packages

setup(
    name='Surface Coil Intensity Correction',
    version='0.5',
    packages=['surface_coil_intensity_correction'],
    # metadata to display on PyPI
    author='Xuan Lei',
    author_email='lei.337@osu.edu',
    description='brightness_correction_map_generator_for_siemens_rawdata(twix)',
    long_description=open('README.md').read(),
    url='',
    classifiers=[
        'License :: all-rights-reserved'
    ],
    python_requires='>=3.8.10',
)
