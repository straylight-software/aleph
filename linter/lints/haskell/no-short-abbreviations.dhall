{- Enforces the Three-Character Rule: no abbreviations less than 4 characters.

This is critical for agent-heavy development where short abbreviations
create exponential confusion across hundreds of contributions.

From the Weyl Standard Haskell style guide:
> If an identifier is 3 characters or less, it's probably too short for production code

Examples:
  - BAD:  cfg, conn, res, req, ctx, idx, mgr, proc, buf
  - GOOD: config, connection, result, request, context, index, manager, process, buffer

Standard exceptions (only in local scope where type makes it unambiguous):
- xs, ys - lists in pure functions
- m, n - indices in array algorithms  
- k, v - key/value in map operations
- f, g - functions in higher-order contexts
-}
let Schema = ../../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "haskell-no-short-abbreviations"
    , language = "haskell"
    , severity = Severity.Warning
    , rule = Schema.Rule::{
      , kind = "variable"
      , regex = Some "^(cfg|conn|res|req|ctx|mgr|proc|buf|ptr|ref|val|num|cnt|tmp|cur|prev|next|init|calc|util|lib|mod|dir|file|path|err|msg|str|arr|vec|map|set|key|src|dst|arg|param|ret|func|meth|obj|inst|ns|pkg|sys|app|prog|db|api|ui|ux|os|cpu|gpu|ram|rom|io|id|uuid|guid|url|uri|ip|ssl|tls|http|https|tcp|udp|dns|ftp|ssh|sql|json|xml|yaml|toml|csv|txt|doc|pdf|png|jpg|gif|svg|html|css|js|ts|jsx|tsx|py|rb|go|rs|java|kt|swift|cpp|c|h|hpp|cs|php|sh|bash|zsh|fish|vim|emacs|git|svn|hg|docker|k8s|aws|gcp|azure)$"
      }
    , message = "ALEPH-H001: Short abbreviation violates Three-Character Rule"
    , note =
        ''
        ## What's wrong?
        Abbreviations less than 4 characters create exponential confusion in agent-heavy codebases.

        From the Weyl Standard Haskell style guide:
        > If an identifier is 3 characters or less, it's probably too short for production code

        ## Examples of violations:
        - `cfg` → use `configuration`
        - `conn` → use `connection`
        - `res` → use `result`
        - `req` → use `request`
        - `ctx` → use `context`
        - `mgr` → use `manager`
        - `proc` → use `process` or `procedure`
        - `buf` → use `buffer`

        ## Standard exceptions (only where type makes it unambiguous):
        - `xs`, `ys` - lists in pure functions
        - `m`, `n` - indices in array algorithms
        - `k`, `v` - key/value in map operations
        - `f`, `g` - functions in higher-order contexts

        ## What can I do to fix this?
        Use full words instead of abbreviations. The extra typing is worth the clarity.

        ```haskell
        -- BAD
        cfg <- loadCfg
        conn <- mkConn cfg
        res <- proc req

        -- GOOD
        configuration <- loadServerConfiguration
        connection <- createDatabaseConnection configuration
        response <- processClientRequest request
        ```
        ''
    , tests =
        { valid = 
            [ "configuration"
            , "connection"
            , "result"
            , "request"
            , "context"
            , "manager"
            , "process"
            , "buffer"
            , "index"
            , "value"
            , "string"
            , "xs"
            , "ys"
            , "f"
            , "g"
            ]
        , invalid = 
            [ "cfg"
            , "conn"
            , "res"
            , "req"
            , "ctx"
            , "mgr"
            , "proc"
            , "buf"
            , "ptr"
            , "tmp"
            ]
        }
    }
