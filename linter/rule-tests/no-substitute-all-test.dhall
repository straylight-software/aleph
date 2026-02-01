{ id = "no-substitute-all"
, valid =
  [ "substitute { src = ./file; }"
  ]
, invalid =
  [ "substituteAll { src = ./file; }"
  ]
}
