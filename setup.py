from setuptools import setup, find_packages
# Using setuptools to define the package configuration
setup(
    name='estsblib',  # The name of the package
    version='0.1',  # The initial version of the package
    description='A demonstration package with various submodules',  # A brief description of the package
    url='',  # The URL for the package's homepage
    author='Ilham Didih',  # The name of the package author
    author_email='ilhamdidih@gmail.com',  # The author's email address
    license='MIT',  # The license under which the package is released
    packages=find_packages(),  # Automatically find and include all packages in the distribution
    zip_safe=False  # Indicates whether the package can be reliably used if installed as a .egg file
)

# NOTE:
# .egg file is a package that contains the package and its dependencies.
# It is used to distribute the package to other computers.
# It is created by running the setup.py script with the build command.
# The .egg file is then installed by running the install command.