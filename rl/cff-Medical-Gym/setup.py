from setuptools import setup

setup(name='gym_medical',
      version='0.0.1',
      install_requires=['gymnasium', 'transformers>=4.8.2', "tokenizers>=0.10.3", "xmltodict"],  # And any other dependencies foo needs
      packages=["data", "gym_medical"]
)
