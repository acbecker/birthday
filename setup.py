try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

DESCRIPTION = "Time Series Analysis of Birthday Data in Python"
LONG_DESCRIPTION = open('README.md').read()
NAME = "birthday"
AUTHOR = "Andrew Becker / Scott Daniel"
AUTHOR_EMAIL = "acbecker@u.washington.edu"
MAINTAINER = "Andrew Becker"
MAINTAINER_EMAIL = "acbecker@u.washington.edu"
DOWNLOAD_URL = 'http://github.com/acbecker/birthday'
URL = DOWNLOAD_URL
LICENSE = 'MIT'
VERSION = '0.0.1'

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['birthday',],
     )
