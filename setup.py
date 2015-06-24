from distutils.core import setup

setup(
    name = 'pyrvm',
    # This makes pyrvm's directory the root (specified by '') instead of the
    # default ('pyrvm')
    package_dir = {'pyrvm': ''},
    packages = ['pyrvm'],
    version = '0.1',
    description = 'Ranking Vector Machines in Python',
    author = 'Robert DiPietro',
    author_email = 'rdipietro@gmail.com',
    url = 'https://github.com/rdipietro/pyrvm',
    download_url = 'https://github.com/rdipietro/pyrvm/tarball/0.1',
    keywords = [],
    classifiers = [],
)
