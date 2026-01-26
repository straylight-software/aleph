//! Dhall → Starlark Transpiler
//!
//! This is a sketch of the transpiler. It's ~150 lines to handle the core cases.
//! The key insight: BUCK files are just data (function calls with kwargs),
//! so transpilation is mechanical.

use dhall::semantics::Nir;
use dhall::syntax::Expr;

/// Convert a Dhall value to Starlark syntax
pub fn to_starlark(value: &Nir) -> String {
    use dhall::semantics::NirKind::*;
    
    match value.kind() {
        // Records become keyword arguments or dicts
        RecordLit(fields) => {
            let pairs: Vec<String> = fields
                .iter()
                .map(|(k, v)| format!("{} = {}", k, to_starlark(v)))
                .collect();
            format!("{{{}}}", pairs.join(", "))
        }
        
        // Lists become lists
        List(items) => {
            let elements: Vec<String> = items
                .iter()
                .map(|v| to_starlark(v))
                .collect();
            format!("[{}]", elements.join(", "))
        }
        
        // Text becomes strings
        TextLit(chunks) => {
            // Handle interpolation if needed
            let s = chunks.to_string();
            format!("\"{}\"", escape_starlark(&s))
        }
        
        // Booleans
        BoolLit(true) => "True".to_string(),
        BoolLit(false) => "False".to_string(),
        
        // Numbers
        NaturalLit(n) => n.to_string(),
        IntegerLit(n) => n.to_string(),
        DoubleLit(n) => format!("{}", n),
        
        // None/Some for Optional
        EmptyOptionalLit(_) => "None".to_string(),
        NEOptionalLit(inner) => to_starlark(inner),
        
        // Unions (enums) - depends on schema
        // For now, just use the variant name as a string
        UnionConstructor(name, _) => format!("\"{}\"", name),
        UnionLit(name, inner, _) => {
            // Could be struct-like or just a tag
            if inner.kind().is_record_lit() {
                format!("{}({})", name, to_starlark(inner))
            } else {
                format!("\"{}\"", name)
            }
        }
        
        // Variables shouldn't appear in normalized output
        Var(_) => panic!("Unexpected variable in normalized Dhall"),
        
        // Everything else
        _ => format!("# TODO: {:?}", value.kind()),
    }
}

fn escape_starlark(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Emit a Buck2 target from a Dhall rule invocation
pub fn emit_target(rule_name: &str, attrs: &Nir) -> String {
    use dhall::semantics::NirKind::*;
    
    match attrs.kind() {
        RecordLit(fields) => {
            let kwargs: Vec<String> = fields
                .iter()
                .map(|(k, v)| format!("    {} = {},", k, to_starlark(v)))
                .collect();
            format!("{}(\n{}\n)", rule_name, kwargs.join("\n"))
        }
        _ => panic!("Target attributes must be a record"),
    }
}

/// Convert a list of targets to a BUCK file
pub fn emit_buck_file(targets: &[(String, Nir)]) -> String {
    let mut lines = vec![
        "# Generated from BUILD.dhall - DO NOT EDIT".to_string(),
        "# Regenerate with: straylight gen".to_string(),
        "".to_string(),
    ];
    
    for (rule_name, attrs) in targets {
        lines.push(emit_target(rule_name, attrs));
        lines.push("".to_string());
    }
    
    lines.join("\n")
}

// -----------------------------------------------------------------------------
// Example usage (pseudocode)
// -----------------------------------------------------------------------------

/*
fn main() -> Result<()> {
    // 1. Parse and evaluate Dhall
    let cx = Ctxt::new();
    let parsed = Parsed::parse_file(Path::new("BUILD.dhall"))?;
    let resolved = parsed.resolve(cx)?;
    let typed = resolved.typecheck(cx)?;
    let normalized = typed.normalize(cx);
    
    // 2. Extract targets from the normalized value
    //    Assumes BUILD.dhall evaluates to a list of { rule : Text, attrs : {...} }
    let targets = extract_targets(normalized.as_nir())?;
    
    // 3. Emit BUCK file
    let buck_content = emit_buck_file(&targets);
    
    // 4. Write to BUCK
    std::fs::write("BUCK", buck_content)?;
    
    Ok(())
}
*/

// -----------------------------------------------------------------------------
// What BUILD.dhall looks like
// -----------------------------------------------------------------------------

/*
-- BUILD.dhall
let DICE = https://straylight.cx/dice/v1/package.dhall sha256:...

let mylib = DICE.rust_library
    { name = "mylib"
    , srcs = [ "src/lib.rs", "src/foo.rs" ]
    , deps = [ "//other:lib" ]
    , edition = DICE.RustEdition.Edition2024
    }

let mybin = DICE.rust_binary
    { name = "mybin"
    , srcs = [ "src/main.rs" ]
    , deps = [ ":mylib" ]
    }

in  [ { rule = "rust_library", attrs = mylib }
    , { rule = "rust_binary", attrs = mybin }
    ]
*/

// -----------------------------------------------------------------------------
// Generated BUCK output
// -----------------------------------------------------------------------------

/*
# Generated from BUILD.dhall - DO NOT EDIT
# Regenerate with: straylight gen

rust_library(
    name = "mylib",
    srcs = ["src/lib.rs", "src/foo.rs"],
    deps = ["//other:lib"],
    edition = "2024",
)

rust_binary(
    name = "mybin",
    srcs = ["src/main.rs"],
    deps = [":mylib"],
)
*/
