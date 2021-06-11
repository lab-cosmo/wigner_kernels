from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

with open('requirements.txt', 'r') as f:
    requirements = [
        line.strip() for line in f if not line.strip().startswith('#')
    ]

extensions = [
    Extension("pytorch_prototype.*", ["code/*.pyx"],
              extra_compile_args=['-O3', '-fopenmp'],
              extra_link_args=['-fopenmp'])
]
setup(
    name='pytorch_prototype',
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=cythonize(extensions),
    zip_safe=False,
)