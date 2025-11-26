from setuptools import setup, find_packages

setup(
    name='claps',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=2',
        'plotly>=6.0.1',
        'pyvista>=0.44.2',
        'rerun-sdk>=0.22.1',
        'scipy>=1.10',
        'torch>=2.4',
        'tqdm>=4.66.6',
    ],
    include_package_data=True,
    zip_safe=False,
)