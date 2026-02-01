{ id = "or-null-fallback"
, valid =
  [ "{ foo ? null }: foo"
  ]
, invalid =
  [ "args.foo or null"
  ]
}
