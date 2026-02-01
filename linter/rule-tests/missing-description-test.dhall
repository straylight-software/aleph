{ id = "missing-description"
, valid =
  [ ''
    stdenv.mkDerivation { meta = { description = "foo"; }; }
    ''
  ]
, invalid =
  [ ''
    stdenv.mkDerivation { meta = { license = lib.licenses.mit; }; }
    ''
  ]
}
