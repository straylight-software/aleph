{ id = "no-heredoc-in-inline-bash"
, valid =
  [ "echo hello"
  ]
, invalid =
  [ ''
    cat <<EOF
    hello
    EOF
    ''
  ]
}
