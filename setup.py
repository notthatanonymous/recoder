from setuptools import setup, find_packages

import recoder


setup(
  name='recsys-recoder',
  version=recoder.__version__,
  install_requires=['torch', 'annoy==1.17.0',
                    'numpy', 'scipy>=1.5.4',
                    'tqdm==4.59.0', 'glog==0.3.1'],
  packages=find_packages(),
  author='Abdallah Moussawi',
  author_email='abdallah.moussawi@gmail.com',
  url='https://github.com/amoussawi/recoder',
  license='MIT'
)
