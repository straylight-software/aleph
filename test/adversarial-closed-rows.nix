# Adversarial test case for nix-compile
# This should trigger "Type error: Attribute set mismatch (closed rows)"

let
  # Function expecting a closed set { a : Int }
  f = { a }: a;

  # Closed set with wrong key
  wrong = {
    b = 1;
  };
in
f wrong
