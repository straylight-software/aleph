-- nix/lib/scripts/fc-init-build.dhall
--
-- Firecracker VM init script - build section
-- Environment variables are injected by render.dhall-with-vars

let buildCmd : Text = env:BUILD_CMD as Text

in ''
# Run build command
echo ":: Running build command..."
${buildCmd}
EXIT=$?
echo ":: Exit code: $EXIT"

# Trigger clean shutdown
echo o >/proc/sysrq-trigger
''
