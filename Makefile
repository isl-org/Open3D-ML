all:

check-style:
	python ./ci/check_style.py

apply-style:
	python ./ci/check_style.py --do_apply_style
