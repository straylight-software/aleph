{ id = "default-nix-in-packages"
, valid =
  [ "hello = callPackage ./hello.nix { };"
  ]
, invalid =
  [ "hello = callPackage ./default.nix { };"
  ]
}
