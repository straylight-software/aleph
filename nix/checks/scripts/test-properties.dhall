-- Dhall template for property test runner
-- Replaces test-properties.bash template with type-safe env var injection

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
