# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

version = '1.0.0'

setup(name='sar_hs_predictor',
      python_requires=">=3.9",
      namespace_packages=["s1tools"],
      version=version,
      description="",
      long_description="""\
""",
      classifiers=[],  # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='',
      author='Manuel GOACOLOU',
      author_email='mgoacolou@cls.fr',
      url='http://www.cls.fr',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      include_package_data=True,
      zip_safe=False,
      # dependency_links = ["."],
      install_requires=['netCDF4',
                        'lxml',
                        'logbook',
                        'numpy',
                        'scipy',
                        'scikit-image',
                        'configobj',
                        'iso8601',
                        'keras',
                        'keras-applications',
                        'keras-preprocessing',
                        'xarray',
                        'dask',
                        'pandas',
                        'pandas-datareader',
                        'pandas-profiling',
                        'tensorflow',
                        'scikit-learn',
                        'xgboost'
                        ],
      entry_points={
          'console_scripts': [
          ]
      },
      )
