-- nix/modules/flake/scripts/lre-buckconfig.dhall
--
-- NativeLink Remote Execution configuration for Buck2
-- Environment variables are injected by render.dhall-with-vars

let port : Text = env:PORT as Text
let instanceName : Text = env:INSTANCE_NAME as Text

in ''

# ────────────────────────────────────────────────────────────────────────
# NativeLink Remote Execution (LRE)
# ────────────────────────────────────────────────────────────────────────
# Usage: lre-start && buck2 build --prefer-remote //:target

[build]
execution_platforms = toolchains//:lre

[buck2_re_client]
engine_address = grpc://127.0.0.1:${port}
cas_address = grpc://127.0.0.1:${port}
action_cache_address = grpc://127.0.0.1:${port}
tls = false
instance_name = ${instanceName}

[buck2_re_client.platform_properties]
OSFamily = linux
container-image = nix-worker
''
