mkdir -p $out/bin
makeWrapper @cloudHypervisorGpu@/bin/cloud-hypervisor-gpu $out/bin/cloud-hypervisor-gpu \
	--set CONFIG_FILE @dhallConfig@ \
	--prefix PATH : @binPath@
