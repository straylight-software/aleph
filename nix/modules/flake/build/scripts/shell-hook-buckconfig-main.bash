# Generate .buckconfig if missing
if [ ! -e ".buckconfig" ]; then
	echo "Generating .buckconfig..."
	cp @buckconfigMainIni@ .buckconfig
	chmod 644 .buckconfig
	echo "Generated .buckconfig"
fi

# Generate .buckroot if missing
if [ ! -e ".buckroot" ]; then
	touch .buckroot
	echo "Generated .buckroot"
fi

# Generate none/BUCK if missing
if [ ! -e "none/BUCK" ]; then
	mkdir -p none
	touch none/BUCK
	echo "Generated none/BUCK"
fi
