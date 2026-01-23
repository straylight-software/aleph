--| DICE Action Types
--|
--| The unit of computation in the build graph.
--| This is what Buck2 runs.

let Toolchain = ./Toolchain.dhall
let Target = ./Target.dhall

let Artifact = Toolchain.Artifact

let ActionCategory =
      < Compile
      | Link
      | Archive
      | Copy
      | Write
      | Run
      | Test
      | Custom : Text
      >

let EnvVar =
      { name : Text
      , value : Text
      }

let Input =
      < Artifact : Artifact
      | Source : Text          -- Source file path
      | Dep : Text             -- Dependency target label
      >

let Output =
      { name : Text
      , binding : Optional Text  -- Environment variable to bind path to
      }

let Action =
      { category : ActionCategory
      , identifier : Text
      , inputs : List Input
      , outputs : List Output
      , command : List Text
      , env : List EnvVar
      , toolchain : Optional Toolchain.Toolchain
      }

-- Action constructors

let compile
    : Text -> List Text -> Text -> Toolchain.Toolchain -> Action
    = \(identifier : Text) ->
      \(srcs : List Text) ->
      \(output : Text) ->
      \(toolchain : Toolchain.Toolchain) ->
        { category = ActionCategory.Compile
        , identifier
        , inputs = List/map Text Input (\(s : Text) -> Input.Source s) srcs
        , outputs = [ { name = output, binding = Some "OUT" } ]
        , command = [] : List Text  -- Filled by rule
        , env = [] : List EnvVar
        , toolchain = Some toolchain
        }

let link
    : Text -> List Artifact -> Text -> Toolchain.Toolchain -> Action
    = \(identifier : Text) ->
      \(objects : List Artifact) ->
      \(output : Text) ->
      \(toolchain : Toolchain.Toolchain) ->
        { category = ActionCategory.Link
        , identifier
        , inputs = List/map Artifact Input (\(a : Artifact) -> Input.Artifact a) objects
        , outputs = [ { name = output, binding = Some "OUT" } ]
        , command = [] : List Text
        , env = [] : List EnvVar
        , toolchain = Some toolchain
        }

let copy
    : Text -> Artifact -> Text -> Action
    = \(identifier : Text) ->
      \(src : Artifact) ->
      \(dst : Text) ->
        { category = ActionCategory.Copy
        , identifier
        , inputs = [ Input.Artifact src ]
        , outputs = [ { name = dst, binding = None Text } ]
        , command = [ "cp", "-r", src.name, dst ]
        , env = [] : List EnvVar
        , toolchain = None Toolchain.Toolchain
        }

let write
    : Text -> Text -> Text -> Action
    = \(identifier : Text) ->
      \(content : Text) ->
      \(output : Text) ->
        { category = ActionCategory.Write
        , identifier
        , inputs = [] : List Input
        , outputs = [ { name = output, binding = None Text } ]
        , command = [] : List Text  -- Content written directly
        , env = [] : List EnvVar
        , toolchain = None Toolchain.Toolchain
        }

let run
    : Text -> List Text -> List EnvVar -> Action
    = \(identifier : Text) ->
      \(command : List Text) ->
      \(env : List EnvVar) ->
        { category = ActionCategory.Run
        , identifier
        , inputs = [] : List Input
        , outputs = [] : List Output
        , command
        , env
        , toolchain = None Toolchain.Toolchain
        }

in  { ActionCategory
    , EnvVar
    , Input
    , Output
    , Action
    , compile
    , link
    , copy
    , write
    , run
    }
