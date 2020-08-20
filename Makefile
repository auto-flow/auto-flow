reinstall:
	pip uninstall auto-flow -y
	rm -rf build dist *.egg-info
	python setup.py install

upload:
	rm -rf build dist *.egg-info
	python setup.py sdist
	twine upload dist/*


hello:
	echo "hello world"

change_pip_source:
	mkdir /root/.pip
	wget https://gitee.com/TQCAI/misc/raw/master/douban_pip_source -O /root/.pip/pip.conf

change_apt_source:
	mv /etc/apt/sources.list /etc/apt/sources.list.bak
	wget https://gitee.com/TQCAI/misc/raw/master/debian_apt_source  -O /etc/apt/sources.list
	apt update -y

install_apt_deps:
	apt install build-essential -y
	apt install swig -y
	apt install libpq-dev -y
	apt install graphviz -y


install_pip_deps:
	pip install Cython==0.29.14
	pip install numpy==1.18.1
	pip install pandas==1.0.1
	pip install scipy==1.3.0
	pip install pyrfr==0.8.0
	pip install ConfigSpaceX==0.4.12
	pip install joblib==0.13.2
	pip install mlxtend==0.17.0
	pip install category_encoders==2.0.0
	pip install lightgbm==2.2.3
	pip install catboost==0.22
	pip install seaborn==0.9.0
	pip install matplotlib==3.0.0
	pip install ruamel.yaml==0.16.0
	pip install scikit_learn==0.22.2
	pip install psutil>=5.5.1
	pip install pynisher>=0.4.1
	pip install imbalanced-learn==0.6.2
	pip install sobol-seq==0.1.2
	pip install lazy-import==0.2.2
	pip install pyDOE==0.3.8
	pip install emcee==3.0.2
	pip install json5==0.9.1
	pip install psycopg2==2.8.5
	pip install peewee==3.13.1
	pip install PyYAML==5.3
	pip install dill==0.3.1.1
	pip install typing-extensions==3.7.4.2
	pip install frozendict==1.2
	pip install scikit-optimize==0.7.4
	pip install hyperopt==0.1.2
	pip install hdfs==2.5.8
	pip install redis==3.4.1
	pip install inflection==0.4.0
	pip install h5py==2.10.0
	pip install datefinder==0.7.0
	pip install tables==3.6.1
	pip install click>=7.0
	pip install ipython
	pip install jupyter
	pip install graphviz
	pip install openpyxl==3.0.3
	pip install requests>=2.21.0
	pip install tabulate