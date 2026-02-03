{- Generate ast-grep configuration

   Usage:
       dhall-to-yaml-ng --file ./sgconfig.dhall
   
   Outputs the sgconfig.yaml content
-}

{ ruleDirs = [ "./rules" ]
, testConfigs = [ { testDir = "./tests" } ]
}
