#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! clap = { version = "4", features = ["string", "env"] }
//! rand = "0.8"
//! rand_chacha = "0.3"
//! serde = { version = "1", features = ["derive"] }
//! serde_json = "1"
//! ```

use clap::{Arg, ArgAction, Command};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::env;

// ============================================================================
// Ground truth types (what we expect the Haskell parser to produce)
// ============================================================================

#[derive(Debug, Serialize)]
struct GroundTruth {
    name: String,
    positional_args: Vec<PositionalTruth>,
    options: Vec<OptionTruth>,
    subcommands: Vec<SubcommandTruth>,
}

#[derive(Debug, Serialize)]
struct PositionalTruth {
    name: String,
    help: String,
    required: bool,
    multiple: bool,
}

#[derive(Debug, Serialize)]
struct OptionTruth {
    short: Option<char>,
    long: Option<String>,
    value_name: Option<String>,
    help: String,
    required: bool,
    takes_value: bool,
    multiple: bool,
    env: Option<String>,
    default: Option<String>,
}

#[derive(Debug, Serialize)]
struct SubcommandTruth {
    name: String,
    about: String,
}

// ============================================================================
// Random generation
// ============================================================================

const ADJECTIVES: &[&str] = &[
    "fast", "slow", "red", "blue", "hidden", "verbose", "quiet", "dry",
    "recursive", "force", "all", "none", "first", "last", "raw", "pretty",
    "json", "yaml", "strict", "loose", "safe", "unsafe", "inline", "external",
];

const NOUNS: &[&str] = &[
    "file", "output", "input", "config", "target", "source", "path", "dir",
    "name", "pattern", "format", "level", "count", "size", "width", "depth",
    "timeout", "limit", "offset", "index", "key", "value", "mode", "type",
];

const VERBS: &[&str] = &[
    "run", "build", "test", "check", "lint", "fmt", "clean", "init", "new",
    "add", "remove", "list", "show", "get", "set", "update", "delete", "sync",
];

const VALUE_NAMES: &[&str] = &[
    "FILE", "PATH", "DIR", "NAME", "PATTERN", "NUM", "COUNT", "SIZE",
    "FORMAT", "LEVEL", "MODE", "TYPE", "KEY", "VALUE", "SECONDS", "BYTES",
    "URL", "HOST", "PORT", "ADDR", "EXPR", "GLOB", "REGEX", "CMD",
];

const POSSIBLE_VALUES: &[&[&str]] = &[
    &["json", "yaml", "toml", "xml"],
    &["debug", "info", "warn", "error"],
    &["auto", "always", "never"],
    &["fast", "balanced", "thorough"],
    &["none", "basic", "full"],
    &["ascii", "utf8", "binary"],
];

const POSITIONAL_NAMES: &[&str] = &[
    "FILE", "PATH", "DIR", "PATTERN", "COMMAND", "TARGET", "SOURCE",
    "INPUT", "OUTPUT", "NAME", "QUERY", "EXPR", "URL", "ARGS",
];

const HELP_TEMPLATES: &[&str] = &[
    "Enable {} mode",
    "Set the {} to use",
    "Specify the {}",
    "Path to the {}",
    "Number of {} to process",
    "Use {} format",
    "Override the default {}",
    "Enable or disable {}",
    "Maximum {} allowed",
    "Minimum {} required",
    "The {} for this operation",
    "Show {} information",
    "Include {} in output",
    "Exclude {} from processing",
    "Filter by {}",
];

fn random_name(rng: &mut impl Rng) -> String {
    if rng.gen_bool(0.6) {
        // adjective-noun style
        format!(
            "{}-{}",
            ADJECTIVES.choose(rng).unwrap(),
            NOUNS.choose(rng).unwrap()
        )
    } else {
        // single word
        NOUNS.choose(rng).unwrap().to_string()
    }
}

fn random_long_name(rng: &mut impl Rng) -> String {
    if rng.gen_bool(0.4) {
        // compound name
        format!(
            "{}-{}",
            ADJECTIVES.choose(rng).unwrap(),
            NOUNS.choose(rng).unwrap()
        )
    } else {
        // single word - prefer nouns for options
        if rng.gen_bool(0.7) {
            NOUNS.choose(rng).unwrap().to_string()
        } else {
            ADJECTIVES.choose(rng).unwrap().to_string()
        }
    }
}

fn random_short(rng: &mut impl Rng, used: &mut Vec<char>) -> Option<char> {
    let candidates: Vec<char> = ('a'..='z')
        .chain('A'..='Z')
        .filter(|c| !used.contains(c))
        .collect();

    if candidates.is_empty() {
        return None;
    }

    let c = *candidates.choose(rng).unwrap();
    used.push(c);
    Some(c)
}

fn random_help(rng: &mut impl Rng) -> String {
    let template = *HELP_TEMPLATES.choose(rng).unwrap();
    let noun = *NOUNS.choose(rng).unwrap();
    template.replace("{}", noun)
}

fn random_value_name(rng: &mut impl Rng) -> String {
    VALUE_NAMES.choose(rng).unwrap().to_string()
}

fn random_env_var(rng: &mut impl Rng) -> String {
    format!(
        "{}_{}",
        NOUNS.choose(rng).unwrap().to_uppercase(),
        NOUNS.choose(rng).unwrap().to_uppercase()
    )
}

fn random_default(rng: &mut impl Rng) -> String {
    match rng.gen_range(0..4) {
        0 => rng.gen_range(0..100).to_string(),
        1 => format!("./{}", NOUNS.choose(rng).unwrap()),
        2 => ["true", "false"].choose(rng).unwrap().to_string(),
        _ => NOUNS.choose(rng).unwrap().to_string(),
    }
}

// ============================================================================
// Clap command generation
// ============================================================================

fn generate_arg(rng: &mut impl Rng, used_shorts: &mut Vec<char>, used_longs: &mut Vec<String>) -> (Arg, OptionTruth) {
    // Generate unique long name
    let mut long_name = random_long_name(rng);
    let mut attempts = 0;
    while used_longs.contains(&long_name) && attempts < 50 {
        long_name = random_long_name(rng);
        attempts += 1;
    }
    used_longs.push(long_name.clone());
    
    let help_text = random_help(rng);

    // Decide characteristics
    let has_short = rng.gen_bool(0.6);
    let takes_value = rng.gen_bool(0.5);
    let required = takes_value && rng.gen_bool(0.2);
    let multiple = takes_value && rng.gen_bool(0.2);
    let has_env = takes_value && rng.gen_bool(0.15);
    let has_default = takes_value && !required && rng.gen_bool(0.2);
    let has_possible_values = takes_value && !multiple && rng.gen_bool(0.15);

    let short_char = if has_short {
        random_short(rng, used_shorts)
    } else {
        None
    };

    let value_name = if takes_value {
        Some(random_value_name(rng))
    } else {
        None
    };

    let env_var = if has_env {
        Some(random_env_var(rng))
    } else {
        None
    };

    let default_val = if has_default {
        Some(random_default(rng))
    } else {
        None
    };

    // Build the clap Arg
    let mut arg = Arg::new(&long_name)
        .long(&long_name)
        .help(&help_text);

    if let Some(c) = short_char {
        arg = arg.short(c);
    }

    if takes_value {
        arg = arg.action(if multiple {
            ArgAction::Append
        } else {
            ArgAction::Set
        });

        if let Some(ref vn) = value_name {
            arg = arg.value_name(vn);
        }

        if required {
            arg = arg.required(true);
        }

        if let Some(ref env) = env_var {
            arg = arg.env(env);
        }

        if let Some(ref def) = default_val {
            arg = arg.default_value(def.clone());
        }
        
        if has_possible_values {
            let values = *POSSIBLE_VALUES.choose(rng).unwrap();
            arg = arg.value_parser(values.to_vec());
        }
    } else {
        arg = arg.action(ArgAction::SetTrue);
    }

    let truth = OptionTruth {
        short: short_char,
        long: Some(long_name),
        value_name,
        help: help_text,
        required,
        takes_value,
        multiple,
        env: env_var,
        default: default_val,
    };

    (arg, truth)
}

fn generate_positional(rng: &mut impl Rng, used_names: &mut Vec<String>) -> (Arg, PositionalTruth) {
    // Generate unique positional name
    let mut name = POSITIONAL_NAMES.choose(rng).unwrap().to_string();
    let mut attempts = 0;
    while used_names.contains(&name) && attempts < 20 {
        name = POSITIONAL_NAMES.choose(rng).unwrap().to_string();
        attempts += 1;
    }
    used_names.push(name.clone());
    
    let help_text = random_help(rng);
    let required = rng.gen_bool(0.6);
    let multiple = !required && rng.gen_bool(0.3); // Only optional args can be multiple
    
    let mut arg = Arg::new(&name)
        .value_name(&name)
        .help(&help_text);
    
    if required {
        arg = arg.required(true);
    }
    
    if multiple {
        arg = arg.action(ArgAction::Append);
    }
    
    let truth = PositionalTruth {
        name: name.clone(),
        help: help_text,
        required,
        multiple,
    };
    
    (arg, truth)
}

fn generate_subcommand(rng: &mut impl Rng) -> (Command, SubcommandTruth) {
    let name = VERBS.choose(rng).unwrap().to_string();
    let about = random_help(rng);

    let cmd = Command::new(&name).about(&about);

    let truth = SubcommandTruth { name, about };

    (cmd, truth)
}

fn generate_command(rng: &mut impl Rng) -> (Command, GroundTruth) {
    let name = random_name(rng);
    let about = random_help(rng);

    let mut cmd = Command::new(&name)
        .about(&about)
        .version("1.0.0")
        .author("Test Author <test@example.com>");

    let mut used_shorts = vec!['h', 'V']; // reserved by clap
    let mut used_longs = vec!["help".to_string(), "version".to_string()]; // reserved by clap
    let mut used_positionals = Vec::new();
    let mut positional_args = Vec::new();
    let mut options = Vec::new();
    let mut subcommands = Vec::new();

    // Maybe generate 0-3 positional arguments (before options)
    if rng.gen_bool(0.5) {
        let num_positionals = rng.gen_range(1..=3);
        for _ in 0..num_positionals {
            let (arg, truth) = generate_positional(rng, &mut used_positionals);
            cmd = cmd.arg(arg);
            positional_args.push(truth);
        }
    }

    // Generate 3-15 options
    let num_args = rng.gen_range(3..=15);
    for _ in 0..num_args {
        let (arg, truth) = generate_arg(rng, &mut used_shorts, &mut used_longs);
        cmd = cmd.arg(arg);
        options.push(truth);
    }

    // Maybe generate 0-5 subcommands
    if rng.gen_bool(0.4) {
        let num_subs = rng.gen_range(1..=5);
        for _ in 0..num_subs {
            let (sub_cmd, truth) = generate_subcommand(rng);
            cmd = cmd.subcommand(sub_cmd);
            subcommands.push(truth);
        }
    }

    let truth = GroundTruth {
        name,
        positional_args,
        options,
        subcommands,
    };

    (cmd, truth)
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse seed from --seed N or use random
    let seed: u64 = args
        .windows(2)
        .find(|w| w[0] == "--seed")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or_else(|| rand::thread_rng().gen());

    // Check for --short flag
    let use_short_help = args.iter().any(|a| a == "--short" || a == "-s");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let (cmd, truth) = generate_command(&mut rng);

    // Render help to string
    let mut help_buf = Vec::new();
    let mut cmd_for_help = cmd.clone();

    if use_short_help {
        cmd_for_help.write_help(&mut help_buf).unwrap();
    } else {
        cmd_for_help.write_long_help(&mut help_buf).unwrap();
    }

    let help_text = String::from_utf8(help_buf).unwrap();

    // Output: first line is JSON ground truth, rest is help text
    println!("{}", serde_json::to_string(&truth).unwrap());
    println!("---");
    print!("{}", help_text);

    // Also output seed to stderr for reproducibility
    eprintln!("seed: {}", seed);
}
