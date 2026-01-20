# config.ncl
# Type-checked configuration for users and machines
#
# Nickel's type system catches errors at evaluation time,
# before any Nix evaluation happens.

let User = String in

let Machine = {
  hostname | String,
  ip | String,
  role | String,
} in

let Config = {
  users | Array User,
  machines | { _ : Machine },
} in

{
  users = ["alice", "bob", "charlie"],

  machines = {
    galois = {
      hostname = "galois",
      ip = "10.0.0.1",
      role = "compute",
    },
    invariant = {
      hostname = "invariant",
      ip = "10.0.0.2",
      role = "storage",
    },
    ultraviolence = {
      hostname = "ultraviolence",
      ip = "10.0.0.3",
      role = "gateway",
    },
  },
} | Config
