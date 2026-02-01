{ id = "rec-anywhere"
, valid =
  [ "let x = 1; in x"
  , "{ a = 1; b = 2; }"
  ]
, invalid =
  [ "rec { a = 1; b = a; }"
  ]
}
