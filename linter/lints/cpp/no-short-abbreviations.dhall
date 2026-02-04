{- Enforces the Three-Letter Rule: no abbreviations less than 4 characters.

This is critical for agent-heavy development where short abbreviations
create exponential confusion across hundreds of contributions.

Examples:
  - BAD:  cfg, conn, res, req, ctx, idx, mgr, proc, buf
  - GOOD: config, connection, result, request, context, index, manager, process, buffer

Common exceptions that are allowed:
  - rx/tx (receive/transmit in networking contexts)
  - i/j/k (loop indices when type makes it unambiguous)
-}
let Schema = ../../schemas/Lint.dhall
let Severity = Schema.Severity

in  { id = "cpp-no-short-abbreviations"
    , language = "cpp"
    , severity = Severity.Error
    , rule = Schema.Rule::{
      , kind = "identifier"
      , regex = Some "^(cfg|conn|res|req|ctx|mgr|proc|buf|ptr|ref|val|num|cnt|tmp|cur|prev|next|init|calc|util|lib|mod|dir|file|path|err|msg|str|arr|vec|map|set|key|val|src|dst|arg|param|ret|func|meth|obj|inst|cls|ns|pkg|sys|app|prog|db|api|ui|ux|os|cpu|gpu|ram|rom|io|id|uuid|guid|url|uri|ip|ssl|tls|http|https|tcp|udp|dns|ftp|ssh|sql|json|xml|yaml|toml|csv|txt|doc|pdf|png|jpg|gif|svg|html|css|js|ts|jsx|tsx|py|rb|go|rs|java|kt|swift|cpp|c|h|hpp|cs|php|sh|bash|zsh|fish|vim|emacs|git|svn|hg|docker|k8s|aws|gcp|azure)$"
      }
    , message = "ALEPH-E001: Short abbreviation violates Three-Letter Rule"
    , note =
        { description = "Abbreviations less than 4 characters create exponential confusion in agent-heavy codebases."
        , examples =
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
        , suggested_fix =
            ''
            Use full words instead of abbreviations. The extra typing is worth the clarity.

            ```cpp
            // BAD
            auto cfg = load_cfg();
            auto conn = db.get_conn();
            auto res = process(req);

            // GOOD
            auto configuration = load_configuration();
            auto connection = database.get_connection();
            auto result = process_request(request);
            ```
            ''
        }
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
            ]
        , extra_invalid = [] : List Text
        }
    }
