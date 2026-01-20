-- config.dhall
-- Type-checked configuration for users and machines

let User = Text

let Machine = { hostname : Text, ip : Text, role : Text }

let Config = { users : List User, machines : { mapKey : Text, mapValue : Machine } }

in  { users = [ "alice", "bob", "charlie" ]
    , machines =
      { galois = { hostname = "galois", ip = "10.0.0.1", role = "compute" }
      , invariant = { hostname = "invariant", ip = "10.0.0.2", role = "storage" }
      , ultraviolence =
        { hostname = "ultraviolence", ip = "10.0.0.3", role = "gateway" }
      }
    }
