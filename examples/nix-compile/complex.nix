{ lib, pkgs }:
let
  # // complex type inference //

  # 1. Polymorphic functions
  # map :: (a -> b) -> [a] -> [b]
  map = f: xs: if xs == [ ] then [ ] else [ (f (builtins.head xs)) ] ++ map f (builtins.tail xs);

  # 2. Higher-order functions
  # apply :: (a -> b) -> a -> b
  apply = f: x: f x;

  # 3. Data structures
  # User :: { name: String, id: Int, active: Bool }
  mkUser = name: id: {
    inherit name id;
    active = true;
  };

  # 4. List processing
  # users :: [User]
  users = [
    (mkUser "alice" 1)
    (mkUser "bob" 2)
  ];

  # 5. Transformation
  # names :: [String]
  names = map (u: u.name) users;

  # 6. Safe primitives (inferred as specific types)
  # path :: Path
  path = ./.;

  # 7. Derivation detection
  # drv :: Derivation
  drv = pkgs.stdenv.mkDerivation {
    name = "test";
    src = ./.;
  };

  # 8. Complex attribute set
  config = {
    services = {
      nginx = {
        enable = true;
        virtualHosts."example.com" = {
          forceSSL = true;
          root = "/var/www";
        };
      };
    };
  };
in
{
  inherit
    users
    names
    drv
    config
    ;
}
