from setuptools import setup, find_packages

setup(
    name='phylogram',
    version='0.0.1',
    url='https://github.com/mypackage.git',
    author='Zhenrui Liao',
    author_email='zhenruiliao@gmail.com',
    description='Minimal package for plotting dendritic morphologies',
    packages=find_packages(),    
    install_requires=[
                      'numpy >= 1.11.1', 
                      'matplotlib >= 1.5.1', 
                      'scipy', 
                      'pandas', 
                      'plotly', 
                      'neuron', 
                      'ipywidgets', 
                      'Bio'],
)
