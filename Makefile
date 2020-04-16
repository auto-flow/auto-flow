reinstall:
	pip uninstall auto-flow -y
	rm -rf build dist *.egg-info
	python setup.py install