chmod:
	@chmod +x ./isearch

clear:
	@find . -name __pycache__ -exec rm -rf {} \;
