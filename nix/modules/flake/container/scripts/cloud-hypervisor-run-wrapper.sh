mkdir -p $out/bin
makeWrapper @cloudHypervisorRun@/bin/cloud-hypervisor-run $out/bin/cloud-hypervisor-run \
	--set CONFIG_FILE @dhallConfig@ \
	--prefix PATH : @binPath@
