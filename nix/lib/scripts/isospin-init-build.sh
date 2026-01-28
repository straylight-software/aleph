# Run build command
echo ":: Running build command..."
@buildCmd@
EXIT=$?
echo ":: Exit code: $EXIT"

# Trigger clean shutdown
echo o >/proc/sysrq-trigger
