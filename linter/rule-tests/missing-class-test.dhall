{ id = "missing-class"
, valid =
  [ ''
    stdenv.mkDerivation { pname = "foo"; }
    ''
  ]
, invalid =
  [ ''
    stdenv.mkDerivation { name = "foo"; }
    ''
  ]
}
