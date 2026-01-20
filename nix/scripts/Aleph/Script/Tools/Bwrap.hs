{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Aleph.Script.Tools.Bwrap
Description : Typed wrapper for bwrap (bubblewrap) - container sandbox

Bubblewrap creates unprivileged sandboxes using Linux namespaces.
This module provides a builder pattern for constructing bwrap invocations
in a type-safe way.

== Key insight: Order matters!

Unlike most CLI tools, bwrap arguments are __ordered__. The sequence of
@--bind@, @--ro-bind@, @--setenv@ etc. is significant. This module uses
a builder pattern to construct the argument list correctly.

@
import Aleph.Script
import qualified Aleph.Script.Tools.Bwrap as Bwrap

main = script $ do
  let sandbox = Bwrap.defaults
        & Bwrap.bind "/tmp/rootfs" "/"
        & Bwrap.dev "/dev"
        & Bwrap.proc "/proc"
        & Bwrap.roBind "/sys" "/sys"
        & Bwrap.tmpfs "/tmp"
        & Bwrap.devBind "/dev/nvidia0" "/dev/nvidia0"
        & Bwrap.setenv "HOME" "/root"
        & Bwrap.chdir "/root"
        & Bwrap.unsharePid
        & Bwrap.dieWithParent

  Bwrap.exec sandbox ["nvidia-smi"]
@
-}
module Aleph.Script.Tools.Bwrap (
    -- * Sandbox builder
    Sandbox (..),
    defaults,
    build,

    -- * Filesystem mounts
    bind,
    bindTry,
    roBind,
    roBindTry,
    devBind,
    devBindTry,
    symlinkBind,
    dev,
    proc,
    tmpfs,
    tmpfsSize,
    mqueue,
    dir,

    -- * Environment
    setenv,
    unsetenv,
    clearenv,
    chdir,
    hostname,

    -- * Namespaces
    unshareAll,
    shareNet,
    unshareUser,
    unshareIpc,
    unsharePid,
    unshareNet,
    unshareUts,
    unshareCgroup,

    -- * Process control
    dieWithParent,
    newSession,
    asPid1,

    -- * Capabilities
    capAdd,
    capDrop,

    -- * Execution
    exec,
    exec_,
    bwrap,
    bwrap_,

    -- * Argument building
    buildArgs,
) where

import Aleph.Script hiding (FilePath)
import Data.Function ((&))
import System.Posix.Process (executeFile)

{- | A bwrap sandbox configuration

Build this using 'defaults' and the various modifier functions.
The 'args' field accumulates arguments in reverse order (for efficiency),
and 'buildArgs' reverses them.
-}
data Sandbox = Sandbox
    { args :: [Text]
    -- ^ Accumulated arguments (reversed)
    }
    deriving (Show, Eq)

-- | Empty sandbox (no configuration)
defaults :: Sandbox
defaults = Sandbox{args = []}

-- | Add raw arguments to the sandbox
addArgs :: [Text] -> Sandbox -> Sandbox
addArgs newArgs s = s{args = reverse newArgs ++ args s}

-- | Build final argument list from sandbox
buildArgs :: Sandbox -> [Text]
buildArgs = reverse . args

-- | Build with command to run
build :: Sandbox -> [Text] -> [Text]
build s cmd = buildArgs s ++ ["--"] ++ cmd

-- ============================================================================
-- Filesystem mounts
-- ============================================================================

{- | Bind mount a host path to a destination

@bind "\/tmp\/rootfs" "\/"@ mounts the rootfs as the sandbox root.
-}
bind :: FilePath -> FilePath -> Sandbox -> Sandbox
bind src dst = addArgs ["--bind", pack src, pack dst]

-- | Bind mount, silently ignore if source doesn't exist
bindTry :: FilePath -> FilePath -> Sandbox -> Sandbox
bindTry src dst = addArgs ["--bind-try", pack src, pack dst]

-- | Read-only bind mount
roBind :: FilePath -> FilePath -> Sandbox -> Sandbox
roBind src dst = addArgs ["--ro-bind", pack src, pack dst]

-- | Read-only bind mount, ignore if missing
roBindTry :: FilePath -> FilePath -> Sandbox -> Sandbox
roBindTry src dst = addArgs ["--ro-bind-try", pack src, pack dst]

-- | Device bind mount (allows device access)
devBind :: FilePath -> FilePath -> Sandbox -> Sandbox
devBind src dst = addArgs ["--dev-bind", pack src, pack dst]

-- | Device bind mount, ignore if missing
devBindTry :: FilePath -> FilePath -> Sandbox -> Sandbox
devBindTry src dst = addArgs ["--dev-bind-try", pack src, pack dst]

-- | Create a symlink in the sandbox
symlinkBind :: FilePath -> FilePath -> Sandbox -> Sandbox
symlinkBind target linkPath = addArgs ["--symlink", pack target, pack linkPath]

-- | Mount a new /dev
dev :: FilePath -> Sandbox -> Sandbox
dev dst = addArgs ["--dev", pack dst]

-- | Mount a new /proc
proc :: FilePath -> Sandbox -> Sandbox
proc dst = addArgs ["--proc", pack dst]

-- | Mount a tmpfs
tmpfs :: FilePath -> Sandbox -> Sandbox
tmpfs dst = addArgs ["--tmpfs", pack dst]

-- | Mount a tmpfs with size limit
tmpfsSize :: Int -> FilePath -> Sandbox -> Sandbox
tmpfsSize bytes dst = addArgs ["--size", pack (show bytes), "--tmpfs", pack dst]

-- | Mount a new mqueue
mqueue :: FilePath -> Sandbox -> Sandbox
mqueue dst = addArgs ["--mqueue", pack dst]

-- | Create a directory
dir :: FilePath -> Sandbox -> Sandbox
dir dst = addArgs ["--dir", pack dst]

-- ============================================================================
-- Environment
-- ============================================================================

-- | Set an environment variable
setenv :: Text -> Text -> Sandbox -> Sandbox
setenv var val = addArgs ["--setenv", var, val]

-- | Unset an environment variable
unsetenv :: Text -> Sandbox -> Sandbox
unsetenv var = addArgs ["--unsetenv", var]

-- | Clear all environment variables
clearenv :: Sandbox -> Sandbox
clearenv = addArgs ["--clearenv"]

-- | Set working directory
chdir :: FilePath -> Sandbox -> Sandbox
chdir path = addArgs ["--chdir", pack path]

-- | Set hostname (requires --unshare-uts)
hostname :: Text -> Sandbox -> Sandbox
hostname name = addArgs ["--hostname", name]

-- ============================================================================
-- Namespaces
-- ============================================================================

-- | Unshare all supported namespaces
unshareAll :: Sandbox -> Sandbox
unshareAll = addArgs ["--unshare-all"]

-- | Keep the network namespace (use with unshareAll)
shareNet :: Sandbox -> Sandbox
shareNet = addArgs ["--share-net"]

-- | Create new user namespace
unshareUser :: Sandbox -> Sandbox
unshareUser = addArgs ["--unshare-user"]

-- | Create new IPC namespace
unshareIpc :: Sandbox -> Sandbox
unshareIpc = addArgs ["--unshare-ipc"]

-- | Create new PID namespace
unsharePid :: Sandbox -> Sandbox
unsharePid = addArgs ["--unshare-pid"]

-- | Create new network namespace
unshareNet :: Sandbox -> Sandbox
unshareNet = addArgs ["--unshare-net"]

-- | Create new UTS namespace
unshareUts :: Sandbox -> Sandbox
unshareUts = addArgs ["--unshare-uts"]

-- | Create new cgroup namespace
unshareCgroup :: Sandbox -> Sandbox
unshareCgroup = addArgs ["--unshare-cgroup"]

-- ============================================================================
-- Process control
-- ============================================================================

-- | Kill child when parent dies
dieWithParent :: Sandbox -> Sandbox
dieWithParent = addArgs ["--die-with-parent"]

-- | Create a new terminal session
newSession :: Sandbox -> Sandbox
newSession = addArgs ["--new-session"]

-- | Don't install PID 1 reaper
asPid1 :: Sandbox -> Sandbox
asPid1 = addArgs ["--as-pid-1"]

-- ============================================================================
-- Capabilities
-- ============================================================================

-- | Add a capability
capAdd :: Text -> Sandbox -> Sandbox
capAdd cap = addArgs ["--cap-add", cap]

-- | Drop a capability
capDrop :: Text -> Sandbox -> Sandbox
capDrop cap = addArgs ["--cap-drop", cap]

-- ============================================================================
-- Execution
-- ============================================================================

{- | Execute bwrap, replacing the current process

This uses 'executeFile' to replace the process, which is the correct
way to hand off to a container. No Haskell code runs after this.
-}
exec :: Sandbox -> [Text] -> Sh ()
exec sandbox cmd =
    liftIO $ executeFile "bwrap" True (map unpack $ build sandbox cmd) Nothing

-- | Execute bwrap, ignoring the exec (for testing)
exec_ :: Sandbox -> [Text] -> Sh ()
exec_ sandbox cmd = run_ "bwrap" (build sandbox cmd)

-- | Run bwrap and capture output (for commands that exit)
bwrap :: Sandbox -> [Text] -> Sh Text
bwrap sandbox cmd = run "bwrap" (build sandbox cmd)

-- | Run bwrap, ignoring output
bwrap_ :: Sandbox -> [Text] -> Sh ()
bwrap_ sandbox cmd = run_ "bwrap" (build sandbox cmd)
