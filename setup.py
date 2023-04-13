from setuptools import find_packages,setup
from typing import List


HYPEN_E_DOT='-e .'
def get_requirements(filepath)->List[str]:
    '''
    function to return list of requirements
    '''
    requirements=[]
    with open(filepath) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements



setup(
    name='data255_dl',
    version='0.0.1',
    author='RK',
    author_email='akashyadav2277@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)