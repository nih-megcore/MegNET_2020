import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MEGnet", 
    version="0.2",
    description="Create ICA features and classify ICA components",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nih-megcore/MegNET_2020",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: UNLICENSE",
        "Operating System :: Linux/Unix",
    ],
    include_package_data=True,
    install_requires=['mne>=1.2','tensorflow'], 
    scripts=['MEGnet/prep_inputs/ICA.py'],
    extras_require={
        "training": ['openpyxl',
            'smote-variants',
'iterative-stratification  @ git+https://github.com/trent-b/iterative-stratification.git@master',
'tensorflow-addons[tensorflow]'],
        "testing": ['pygit2'],}
)
