{ id = "missing-meta"
, valid =
  [ ''
    stdenv.mkDerivation { meta = { description = "foo"; }; }
    ''
  ]
, invalid =
  [ ''
    stdenv.mkDerivation { name = "foo"; }
    ''
  ]
}
