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

install_all:
	pip install Cython
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
	pip install psutil
	pip install pynisher>=0.4.1
	pip install imbalanced-learn==0.6.2
	pip install sobol_seq
	pip install lazy_import
	pip install pyDOE
	pip install emcee
	pip install json5
	pip install peewee
	pip install pyyaml
	pip install dill
	pip install typing_extensions
	pip install frozendict
	pip install scikit-optimize==0.7.4
	pip install hyperopt==0.1.2
	pip install hdfs==2.5.8
	pip install redis
	pip install gensim
	pip install inflection
	pip install h5py
	pip install datefinder
	pip install tables
	pip install click