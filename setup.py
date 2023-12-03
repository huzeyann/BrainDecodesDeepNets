import setuptools

# with open('requirements.txt') as f:
#     required = f.read().splitlines()

setuptools.setup(
    name="brainnet",
    version="0.1.0",
    # url="https://github.com/kiwidamien/roman",
    author="Huzheng Yang",
    author_email="huze.yann@gmail.com",
    description="Brain Decodes Deep Nets",
    # long_description=open('DESCRIPTION.rst').read(),
    packages=setuptools.find_packages(),
    # install_requires=required,
    # classifiers=[
    #     'Programming Language :: Python',
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.6',
    # ],
    # include_package_data=True,
    # package_data={'': ['roi_masks/*.npy']},
)