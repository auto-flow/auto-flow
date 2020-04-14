reinstall:
	pip uninstall HyperFlow -y
	rm -rf build dist *.egg-info
	python setup.py install