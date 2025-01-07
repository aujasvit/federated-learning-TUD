from setuptools import setup, find_packages

setup(
   name='froodo',
   version='1.0',
   description='A useful module',
   author='Jonathan Stieber',
   author_email='foomail@foo.example',
   packages= find_packages(),  #same as name
   install_requires=[], #external packages as dependencies
)

