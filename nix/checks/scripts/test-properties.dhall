-- nix/checks/scripts/test-properties.dhall
--
-- Test script for property tests
-- Environment variables are injected by render.dhall-with-vars

let report : Text = env:REPORT as Text
let resultMessage : Text = env:RESULT_MESSAGE as Text
let touchOut : Text = env:TOUCH_OUT as Text

in ''
#!/usr/bin/env bash
# Run property tests

echo "${report}"
echo "${resultMessage}"
${touchOut}
''
