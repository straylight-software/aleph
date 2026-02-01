{ id = "rec-in-derivation"
, valid =
  [ ''
    stdenv.mkDerivation (finalAttrs: { name = "foo"; })
    ''
  , ''
    stdenv.mkDerivation { name = "foo"; }
    ''
  ]
, invalid =
  [ ''
    stdenv.mkDerivation rec { name = "foo"; }
    ''
  ]
}
