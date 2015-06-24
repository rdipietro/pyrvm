from distutils.core import setup

setup(
    name = 'pyrvm',
    # This makes pyrvm's directory the root (specified by '') instead of the
    # default ('pyrvm')
    package_dir = {'pyrvm': ''},
    packages = ['pyrvm'],
    version = '0.12',
    description = 'Ranking Vector Machines in Python',
    author = 'Robert DiPietro',
    author_email = 'rdipietro@gmail.com',
    url = 'https://github.com/rdipietro/pyrvm',
    download_url = 'https://github.com/rdipietro/pyrvm/tarball/0.12',
    install_requires = ['numpy', 'scikit-learn', 'pulp'],
    keywords = [],
    classifiers = [],
)
