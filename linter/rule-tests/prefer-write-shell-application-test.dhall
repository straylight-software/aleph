{ id = "prefer-write-shell-application"
, valid =
  [ ''
    aleph.writeShellApplication { name = "foo"; text = "echo hi"; }
    ''
  ]
, invalid =
  [ ''
    writeShellScript "foo" "echo hi"
    ''
  , ''
    writeShellScriptBin "foo" "echo hi"
    ''
  ]
}
