-- Drv/Src.dhall
-- Source fetchers. Pure data.

let Src
    : Type
    = < GitHub : { owner : Text, repo : Text, rev : Text, hash : Text }
      | Url : { url : Text, hash : Text }
      | Local : { path : Text }
      | None
      >

let github =
      \(owner : Text) ->
      \(repo : Text) ->
      \(rev : Text) ->
      \(hash : Text) ->
        Src.GitHub { owner, repo, rev, hash }

let url = \(u : Text) -> \(hash : Text) -> Src.Url { url = u, hash }

let local = \(path : Text) -> Src.Local { path }

let none = Src.None

in  { Src, github, url, local, none }
