# // nix-compile // reference //

## // types //

| Type | Description | Example |
|------|-------------|---------|
| `Int` | Integer numbers | `42` |
| `Float` | Floating point numbers | `3.14` |
| `Bool` | Boolean values | `true` |
| `String` | Text strings | `"hello"` |
| `Path` | File or directory paths | `./foo` |
| `Null` | The null value | `null` |
| `[a]` | List of type `a` | `[ 1 2 3 ]` |
| `{}` | Attribute set | `{ x = 1; }` |
| `a -> b` | Function from `a` to `b` | `x: x + 1` |
| `Derivation` | A Nix derivation | `pkgs.hello` |
| `Any` | Unknown/Top type | (unconstrained) |

## // bash builtins //

The following commands are known to `nix-compile` and used for type inference of arguments.

### Core Utilities

| Command | Known Flags |
|---------|-------------|
| `mkdir` | `-m` (Mode) |
| `chmod` | (Positional: Mode, File) |
| `chown` | (Positional: Owner, File) |
| `head` | `-n` (Lines), `-c` (Bytes) |
| `tail` | `-n` (Lines), `-c` (Bytes) |
| `split` | `-n`, `-l`, `-b`, `-a` |
| `dd` | `bs`, `count`, `skip`, `seek`, `if`, `of` |
| `sleep` | (Positional: Seconds) |
| `timeout` | `-s` (Signal), `-k` (Kill after) |

### Network

| Command | Known Flags |
|---------|-------------|
| `curl` | `--connect-timeout`, `--max-time`, `-o`, `-H`, `-d`, `-u`, `-X` |
| `wget` | `-O` (Output), `-t` (Retries), `-T` (Timeout) |
| `nc` | `-w` (Timeout), `-p` (Port) |
| `ssh` | `-p` (Port), `-i` (Identity), `-F` (Config), `-l` (Login) |
| `scp` | `-P` (Port), `-i` (Identity), `-F` (Config), `-l` (Limit) |
| `rsync` | `--timeout`, `--port`, `--bwlimit`, `--max-size` |

### Tools

| Command | Known Flags |
|---------|-------------|
| `jq` | `--indent`, `-r` (Raw), `-e` (Exit code), `-s` (Slurp), `-c` (Compact) |
| `grep` | `-m` (Max), `-A/B/C` (Context), `-e` (Pattern), `-f` (File) |
| `find` | `-maxdepth`, `-mindepth`, `-mtime`, `-size`, `-name`, `-type` |
| `xargs` | `-n` (Max args), `-P` (Max procs), `-d` (Delimiter) |
| `parallel` | `-j` (Jobs), `--delay`, `--timeout`, `--retries` |
| `nix` | `--max-jobs`, `--cores` |

## // whitelisted commands //

These commands can be used without absolute store paths (assumed to be in PATH):

*   `cp`, `mv`, `rm`, `mkdir`, `rmdir`, `ln`, `ls`
*   `cat`, `head`, `tail`, `sort`, `uniq`, `wc`, `tr`, `cut`, `tee`
*   `grep`, `sed`, `awk`, `find`, `xargs`
*   `mktemp`, `realpath`, `dirname`, `basename`, `readlink`
*   `env`, `which`, `chmod`, `chown`, `chgrp`, `touch`
*   `sleep`, `clear`, `nproc`, `free`, `udhcpc`
*   `runHook`
*   `makeWrapper`, `wrapProgram`
*   `fakeroot`, `tune2fs`, `sha256sum`, `file`, `patchelf`, `unzip`, `tar`, `gzip`
*   `curl`, `date`, `git`

## // error codes //

### ALEPH-N001: with expression
The `with` construct obscures variable scope and makes static analysis impossible. Use explicit attribute access or `inherit`.

### ALEPH-N002: rec set
Recursive sets can lead to infinite recursion and complicate type inference. Use `let` bindings for mutual recursion.

### Bare Command
A command was invoked without an absolute path. In hermetic builds, all dependencies should be referenced via store paths (e.g. `${pkgs.curl}/bin/curl`). Exceptions are made for standard utilities (see Whitelist).
