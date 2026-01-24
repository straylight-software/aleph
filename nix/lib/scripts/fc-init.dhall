-- nix/lib/scripts/fc-init.dhall
--
-- Firecracker VM init script
-- Environment variables are injected by render.dhall-with-vars

let envExports : Text = env:ENV_EXPORTS as Text
let baseInit : Text = env:BASE_INIT as Text
let networkSetup : Text = env:NETWORK_SETUP as Text
let buildSection : Text = env:BUILD_SECTION as Text
let interactiveSection : Text = env:INTERACTIVE_SECTION as Text

in ''
#!/bin/sh
set -e

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
${envExports}

${baseInit}

hostname builder

${networkSetup}

${buildSection}

# Drop to interactive shell if no build command
${interactiveSection}
''
