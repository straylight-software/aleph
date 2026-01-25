# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                    NIXPKGS LIB REIMPLEMENTED VIA WEYL PRELUDE              ║
# ║                                                                            ║
# ║  A drop-in replacement for nixpkgs/lib that uses the Aleph Prelude          ║
# ║  for all core functional primitives.                                       ║
# ║                                                                            ║
# ║  The legacy lib is a sprawling mess of inconsistent naming:                ║
# ║    - lib.reverseList vs lib.concatLists (why not reverseLists?)            ║
# ║    - lib.hasPrefix vs lib.elem (prefix on what? elem of what?)             ║
# ║    - lib.foldl' vs lib.foldr (prime? really?)                              ║
# ║    - lib.mapAttrs vs lib.concatMapStrings (Attrs vs Strings suffix)        ║
# ║                                                                            ║
# ║  The Aleph Prelude brings Haskell-style consistency:                        ║
# ║    - Lisp-case naming (map-attrs, fold-right, starts-with)                 ║
# ║    - Predictable argument order (data last for currying)                   ║
# ║    - Type signatures in comments                                           ║
# ║                                                                            ║
# ║  This shim provides backward compatibility while using the prelude         ║
# ║  internally. New code should use the prelude directly.                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝
{ prelude }:

let
  P = prelude;

  # ════════════════════════════════════════════════════════════════════════════
  # TRIVIAL - Core combinators and primitives
  # ════════════════════════════════════════════════════════════════════════════
  trivial = {
    # The Aleph Prelude provides these directly
    inherit (P) id;
    inherit (P) const;
    inherit (P) flip;

    # pipe :: a -> [(a -> b)] -> b
    # Legacy takes a value and list of functions
    # Prelude pipe is binary, so we fold
    pipe = val: fns: P.fold (x: f: f x) val fns;

    # concat :: [a] -> [a] -> [a]
    concat = x: y: x ++ y;

    # Boolean operations - Prelude uses and/or, legacy uses quoted "or"
    "or" = P.or;
    inherit (P) and;
    xor = a: b: (!a) != (!b);

    # Bitwise - direct from builtins
    inherit (builtins) bitAnd bitOr bitXor;
    bitNot = builtins.sub (-1);

    # Conversions
    boolToString = b: if b then "true" else "false";
    boolToYesNo = b: if b then "yes" else "no";

    # mergeAttrs :: Attrs -> Attrs -> Attrs (shallow merge, right wins)
    mergeAttrs = x: y: x // y;

    # defaultTo :: a -> Maybe a -> a (Prelude: from-maybe)
    defaultTo = P.from-maybe;

    # mapNullable :: (a -> b) -> Maybe a -> Maybe b
    mapNullable = f: a: if a == null then null else f a;

    # Arithmetic
    inherit (P) min;
    inherit (P) max;
    inherit (P) mod;
    inherit (P) compare;
    inherit (P) add;
    inherit (P) sub;
    lessThan = P.lt;

    # Sequences
    inherit (builtins) seq deepSeq genericClosure;

    # Type checks - Prelude uses is-* naming
    isFloat = P.is-float;
    isBool = P.is-bool;
    isInt = P.is-int;
    isFunction = P.is-function;
    pathExists = P.path-exists;
    readFile = P.read-file;

    # JSON/TOML
    importJSON = path: P.from-json (P.read-file path);
    importTOML = path: builtins.fromTOML (P.read-file path);

    # Function metadata
    functionArgs =
      f:
      if f ? __functor then
        f.__functionArgs or (trivial.functionArgs (f.__functor f))
      else
        builtins.functionArgs f;

    setFunctionArgs = f: args: {
      __functor = _self: f;
      __functionArgs = args;
    };

    mirrorFunctionArgs = f: g: trivial.setFunctionArgs g (trivial.functionArgs f);
    toFunction = v: if P.is-function v then v else _: v;

    # Warnings/errors
    warn = builtins.warn or (msg: v: builtins.trace "warning: ${msg}" v);
    warnIf = cond: msg: if cond then trivial.warn msg else P.id;
    warnIfNot = cond: msg: if cond then P.id else trivial.warn msg;
    throwIf = cond: msg: if cond then throw msg else P.id;
    throwIfNot = cond: msg: if cond then P.id else throw msg;
    info = msg: builtins.trace "INFO: ${msg}";
    showWarnings = warnings: res: P.fold-right (w: x: trivial.warn w x) res warnings;

    # Hex conversion
    toHexString =
      let
        hex-digits = {
          "10" = "A";
          "11" = "B";
          "12" = "C";
          "13" = "D";
          "14" = "E";
          "15" = "F";
        };
        to-hex-digit = d: if d < 10 then toString d else hex-digits.${toString d};
      in
      i: P.join "" (P.map to-hex-digit (trivial.toBaseDigits 16 i));

    toBaseDigits =
      base: i:
      let
        go =
          i:
          if i < base then
            [ i ]
          else
            let
              r = i - ((i / base) * base);
              q = (i - r) / base;
            in
            [ r ] ++ go q;
      in
      P.reverse (go i);

    fromHexString =
      str:
      let
        match = builtins.match "(0x)?([0-7]?[0-9A-Fa-f]{1,15})" str;
      in
      if match != null then
        (builtins.fromTOML "v=0x${builtins.elemAt match 1}").v
      else
        throw "Invalid hex string: ${str}";

    # Version info (stubs - would need actual nixpkgs)
    version = "24.11";
    release = "24.11";
    versionSuffix = "";
    oldestSupportedRelease = 2411;
    oldestSupportedReleaseIsAtLeast = release: release <= 2411;
    inNixShell = builtins.getEnv "IN_NIX_SHELL" != "";
    inPureEvalMode = !builtins ? currentSystem;
  };

  # ════════════════════════════════════════════════════════════════════════════
  # LISTS - All list operations via Prelude
  # ════════════════════════════════════════════════════════════════════════════
  lists = {
    # Direct mappings from Prelude
    inherit (P) map;
    inherit (P) filter;
    inherit (P) head;
    inherit (P) tail;
    inherit (P) init;
    inherit (P) last;
    inherit (P) take;
    inherit (P) drop;
    inherit (P) length;
    inherit (P) elem;
    inherit (P) sort;
    inherit (P) unique;
    inherit (P) flatten;
    inherit (P) range;
    inherit (P) replicate;
    inherit (P) partition;

    # Renamed for legacy compat
    foldr = P.fold-right;
    inherit (builtins) foldl;
    foldl' = P.fold; # Prelude fold is strict
    reverseList = P.reverse;
    concatLists = P.concat;
    concatMap = P.concat-map;
    zipLists = P.zip;
    zipListsWith = P.zip-with;
    sortOn = P.sort-on;
    groupBy = P.group-by;

    # singleton :: a -> [a]
    singleton = x: [ x ];

    # forEach :: [a] -> (a -> b) -> [b] (flip of map)
    forEach = xs: f: P.map f xs;

    # imap0 :: (Int -> a -> b) -> [a] -> [b]
    imap0 = f: xs: P.zip-with f (P.range 0 (P.length xs - 1)) xs;

    # imap1 :: (Int -> a -> b) -> [a] -> [b]
    imap1 = f: xs: P.zip-with f (P.range 1 (P.length xs)) xs;

    # ifilter0 :: (Int -> a -> Bool) -> [a] -> [a]
    ifilter0 =
      f: xs:
      P.map (pair: pair.snd) (
        P.filter (pair: f pair.fst pair.snd) (
          lists.imap0 (i: x: {
            fst = i;
            snd = x;
          }) xs
        )
      );

    # any :: (a -> Bool) -> [a] -> Bool
    inherit (P) any;

    # all :: (a -> Bool) -> [a] -> Bool
    inherit (P) all;

    # count :: (a -> Bool) -> [a] -> Int
    count = pred: xs: P.length (P.filter pred xs);

    # optional :: Bool -> a -> [a]
    optional = cond: val: if cond then [ val ] else [ ];

    # optionals :: Bool -> [a] -> [a]
    optionals = cond: vals: if cond then vals else [ ];

    # toList :: a -> [a]
    toList = x: if P.is-list x then x else [ x ];

    # findFirst :: (a -> Bool) -> a -> [a] -> a
    findFirst =
      pred: default: xs:
      let
        found = P.filter pred xs;
      in
      if found == [ ] then default else P.head found;

    # findSingle :: (a -> Bool) -> a -> a -> [a] -> a
    findSingle =
      pred: none: multiple: xs:
      let
        found = P.filter pred xs;
        len = P.length found;
      in
      if len == 0 then
        none
      else if len == 1 then
        P.head found
      else
        multiple;

    # remove :: a -> [a] -> [a]
    remove = x: P.filter (y: y != x);

    # subtractLists :: [a] -> [a] -> [a]
    subtractLists = a: b: P.filter (x: !(P.elem x a)) b;

    # intersectLists :: [a] -> [a] -> [a]
    intersectLists = a: b: P.filter (x: P.elem x b) a;

    # mutuallyExclusive :: [a] -> [a] -> Bool
    mutuallyExclusive = a: b: lists.intersectLists a b == [ ];

    # crossLists :: (a -> b -> c) -> [[a]] -> [[b]] -> [c]
    crossLists =
      f: xs: ys:
      P.concat-map (x: P.map (y: f x y) ys) xs;

    # genList :: (Int -> a) -> Int -> [a]
    genList = f: n: P.map f (P.range 0 (n - 1));

    # sublist :: Int -> Int -> [a] -> [a]
    sublist =
      start: count: xs:
      P.take count (P.drop start xs);

    # takeEnd :: Int -> [a] -> [a]
    takeEnd = n: xs: P.drop (P.length xs - n) xs;

    # dropEnd :: Int -> [a] -> [a]
    dropEnd = n: xs: P.take (P.length xs - n) xs;

    # compareLists :: (a -> a -> Int) -> [a] -> [a] -> Int
    compareLists =
      cmp: a: b:
      if a == [ ] && b == [ ] then
        0
      else if a == [ ] then
        -1
      else if b == [ ] then
        1
      else
        let
          c = cmp (P.head a) (P.head b);
        in
        if c != 0 then c else lists.compareLists cmp (P.tail a) (P.tail b);

    # naturalSort :: [String] -> [String]
    naturalSort = builtins.sort (a: b: (builtins.compareVersions a b) < 0);

    # allUnique :: [a] -> Bool
    allUnique = xs: P.length xs == P.length (P.unique xs);

    # elemAt :: [a] -> Int -> a
    elemAt = xs: n: builtins.elemAt xs n;

    # isList :: a -> Bool
    isList = P.is-list;

    # fold :: (a -> b -> b) -> b -> [a] -> b (legacy name for foldr)
    fold = P.fold-right;

    # New additions - Prelude has these, legacy doesn't!
    chunksOf = P.chunks-of;
    splitAt = n: xs: {
      fst = P.take n xs;
      snd = P.drop n xs;
    };
    inherit (P) break;
    findIndex = P.find-index;
    findIndices = P.find-indices;
    elemIndex = P.elem-index;
    elemIndices = P.elem-indices;
    updateAt = P.update-at;
    setAt = P.set-at;
    insertAt = P.insert-at;
    removeAt = P.remove-at;
    nubOn = P.unique-by;
    minimumBy = P.minimum-by;
    maximumBy = P.maximum-by;
  };

  # ════════════════════════════════════════════════════════════════════════════
  # ATTRSETS - All attribute set operations via Prelude
  # ════════════════════════════════════════════════════════════════════════════
  attrsets = {
    # Direct mappings
    mapAttrs = P.map-attrs;
    mapAttrs' = P.map-attrs';
    filterAttrs = P.filter-attrs;
    attrNames = P.keys;
    attrValues = P.values;
    listToAttrs = P.from-list;
    attrsToList = P.to-list;
    genAttrs = P.gen-attrs;
    intersectAttrs = P.intersect;
    recursiveUpdate = P.merge;

    # hasAttr :: String -> Attrs -> Bool
    hasAttr = P.has;

    # getAttr :: String -> Attrs -> a
    getAttr = P.get';

    # isAttrs :: a -> Bool
    isAttrs = P.is-attrs;

    # attrByPath :: [String] -> a -> Attrs -> a
    attrByPath = P.get;

    # setAttrByPath :: [String] -> a -> Attrs
    setAttrByPath = P.set;

    # hasAttrByPath :: [String] -> Attrs -> Bool
    hasAttrByPath =
      path: attrs:
      let
        go =
          p: a:
          if p == [ ] then
            true
          else if P.is-attrs a && P.has (P.head p) a then
            go (P.tail p) a.${P.head p}
          else
            false;
      in
      go path attrs;

    # getAttrFromPath :: [String] -> Attrs -> a
    getAttrFromPath = path: attrs: P.get path (throw "attribute not found") attrs;

    # removeAttrs :: Attrs -> [String] -> Attrs
    # Note: legacy has attrs first, then names
    removeAttrs = attrs: names: P.remove names attrs;

    # attrVals :: [String] -> Attrs -> [a]
    attrVals = names: attrs: P.map (n: attrs.${n}) names;

    # getAttrs :: [String] -> Attrs -> Attrs
    getAttrs =
      names: attrs:
      P.from-list (
        P.map (n: {
          name = n;
          value = attrs.${n};
        }) (P.filter (n: P.has n attrs) names)
      );

    # catAttrs :: String -> [Attrs] -> [a]
    # NOTE: `or null` here is intentional - this IS the safe accessor pattern.
    # We extract the attribute from each attrset (null if missing), then
    # cat-maybes filters out the nulls. This matches nixpkgs lib.catAttrs.
    catAttrs = name: xs: P.cat-maybes (P.map (x: x.${name} or null) xs);

    # filterAttrsRecursive :: (String -> a -> Bool) -> Attrs -> Attrs
    filterAttrsRecursive =
      pred: attrs:
      P.map-attrs (
        _name: value: if P.is-attrs value then attrsets.filterAttrsRecursive pred value else value
      ) (P.filter-attrs pred attrs);

    # foldlAttrs :: (b -> String -> a -> b) -> b -> Attrs -> b
    foldlAttrs =
      f: init: attrs:
      P.fold (acc: pair: f acc pair.name pair.value) init (P.to-list attrs);

    # foldAttrs :: (a -> b -> b) -> b -> [Attrs] -> Attrs
    foldAttrs =
      f: init: xs:
      P.fold-right (
        attrs: acc: P.map-attrs (name: value: f value (acc.${name} or init)) attrs // acc
      ) { } xs;

    # collect :: (a -> Bool) -> Attrs -> [a]
    inherit (P) collect;

    # nameValuePair :: String -> a -> { name: String, value: a }
    nameValuePair = name: value: { inherit name value; };

    # mapAttrsToList :: (String -> a -> b) -> Attrs -> [b]
    mapAttrsToList = P.map-to-list;

    # concatMapAttrs :: (String -> a -> Attrs) -> Attrs -> Attrs
    concatMapAttrs = f: attrs: P.merge-all (P.map-to-list f attrs);

    # mapAttrsRecursive :: ([String] -> a -> b) -> Attrs -> Attrs
    mapAttrsRecursive =
      f: attrs:
      let
        go =
          path: a:
          P.map-attrs (
            name: value: if P.is-attrs value then go (path ++ [ name ]) value else f (path ++ [ name ]) value
          ) a;
      in
      go [ ] attrs;

    # mapAttrsRecursiveCond :: (Attrs -> Bool) -> ([String] -> a -> b) -> Attrs -> Attrs
    mapAttrsRecursiveCond =
      cond: f: attrs:
      let
        go =
          path: a:
          P.map-attrs (
            name: value:
            if P.is-attrs value && cond value then go (path ++ [ name ]) value else f (path ++ [ name ]) value
          ) a;
      in
      go [ ] attrs;

    # optionalAttrs :: Bool -> Attrs -> Attrs
    optionalAttrs = P.when-attr;

    # zipAttrsWithNames :: [String] -> (String -> [a] -> b) -> [Attrs] -> Attrs
    # NOTE: `or null` here is intentional - not every set has every attribute.
    # The function f receives the list of values (with nulls for missing attrs)
    # and decides how to combine them. This matches nixpkgs lib.zipAttrsWithNames.
    zipAttrsWithNames =
      names: f: sets:
      P.from-list (
        P.map (name: {
          inherit name;
          value = f name (P.map (s: s.${name} or null) sets);
        }) names
      );

    # zipAttrsWith :: (String -> [a] -> b) -> [Attrs] -> Attrs
    zipAttrsWith =
      f: sets:
      let
        names = P.unique (P.concat-map P.keys sets);
      in
      attrsets.zipAttrsWithNames names f sets;

    # zipAttrs :: [Attrs] -> Attrs
    zipAttrs = builtins.zipAttrsWith (_: P.cat-maybes);

    # recursiveUpdateUntil :: (path -> a -> b -> Bool) -> Attrs -> Attrs -> Attrs
    recursiveUpdateUntil =
      pred: lhs: rhs:
      let
        go =
          path: l: r:
          if !(P.is-attrs l && P.is-attrs r) || pred path l r then
            r
          else
            P.map-attrs (name: lv: if P.has name r then go (path ++ [ name ]) lv r.${name} else lv) l
            // P.filter-attrs (name: _: !(P.has name l)) r;
      in
      go [ ] lhs rhs;

    # mergeAttrsList :: [Attrs] -> Attrs
    mergeAttrsList = P.merge-all;

    # overrideExisting :: Attrs -> Attrs -> Attrs
    overrideExisting = old: new: P.map-attrs (name: value: new.${name} or value) old;

    # showAttrPath :: [String] -> String
    showAttrPath = P.join ".";

    # isDerivation :: a -> Bool
    isDerivation = x: P.is-attrs x && x ? type && x.type == "derivation";

    # toDerivation :: Path -> Derivation
    toDerivation = path: {
      type = "derivation";
      name = builtins.baseNameOf path;
      outPath = path;
      outputs = [ "out" ];
    };

    # getOutput :: String -> Derivation -> String
    getOutput = output: drv: drv.${output} or drv.out or drv.outPath;

    getBin = attrsets.getOutput "bin";
    getLib = attrsets.getOutput "lib";
    getDev = attrsets.getOutput "dev";
    getMan = attrsets.getOutput "man";

    # recurseIntoAttrs :: Attrs -> Attrs
    recurseIntoAttrs = P.recurse-into-attrs;

    # dontRecurseIntoAttrs :: Attrs -> Attrs
    dontRecurseIntoAttrs = attrs: attrs // { recurseForDerivations = false; };

    # New additions - Prelude has these, legacy doesn't!
    updateAtPath = P.update-at-path;
    setAtPath = P.set-at-path;
    inherit (P) modify;
    renameKey = P.rename-key;
    inherit (P) pick;
    inherit (P) omit;
    inherit (P) defaults;
    inherit (P) invert;

    # cartesianProduct :: Attrs -> [Attrs]
    cartesianProduct =
      attrs:
      let
        names = P.keys attrs;
        go =
          ns:
          if ns == [ ] then
            [ { } ]
          else
            let
              name = P.head ns;
              rest = go (P.tail ns);
            in
            P.concat-map (r: P.map (v: r // { ${name} = v; }) attrs.${name}) rest;
      in
      go names;

    # mapCartesianProduct :: (Attrs -> a) -> Attrs -> [a]
    mapCartesianProduct = f: attrs: P.map f (attrsets.cartesianProduct attrs);
  };

  # ════════════════════════════════════════════════════════════════════════════
  # STRINGS - All string operations via Prelude
  # ════════════════════════════════════════════════════════════════════════════
  strings = {
    # Direct mappings
    stringLength = P.string-length;
    inherit (P) substring;
    splitString = P.split;
    replaceStrings = P.replace;
    hasPrefix = P.starts-with;
    hasSuffix = P.ends-with;
    hasInfix = P.contains;
    toLower = P.to-lower;
    toUpper = P.to-upper;
    inherit (P) trim;
    inherit (P) escape;
    escapeShellArg = P.escape-shell;

    # isString :: a -> Bool
    isString = P.is-string;

    # typeOf :: a -> String
    typeOf = P.typeof;

    # concatStrings :: [String] -> String
    concatStrings = P.join "";

    # concatStringsSep :: String -> [String] -> String
    concatStringsSep = P.join;

    # concatMapStrings :: (a -> String) -> [a] -> String
    concatMapStrings = f: xs: P.join "" (P.map f xs);

    # concatMapStringsSep :: String -> (a -> String) -> [a] -> String
    concatMapStringsSep =
      sep: f: xs:
      P.join sep (P.map f xs);

    # concatImapStrings :: (Int -> a -> String) -> [a] -> String
    concatImapStrings = f: xs: P.join "" (lists.imap1 f xs);

    # concatImapStringsSep :: String -> (Int -> a -> String) -> [a] -> String
    concatImapStringsSep =
      sep: f: xs:
      P.join sep (lists.imap1 f xs);

    # concatLines :: [String] -> String
    concatLines = P.unlines;

    # intersperse :: a -> [a] -> [a]
    inherit (P) intersperse;

    # optionalString :: Bool -> String -> String
    optionalString = P.when-str;

    # removePrefix :: String -> String -> String
    removePrefix = P.remove-prefix;

    # removeSuffix :: String -> String -> String
    removeSuffix = P.remove-suffix;

    # stringToCharacters :: String -> [String]
    stringToCharacters = str: P.map (i: P.substring i 1 str) (P.range 0 (P.string-length str - 1));

    # stringAsChars :: (String -> String) -> String -> String
    stringAsChars = f: str: P.join "" (P.map f (strings.stringToCharacters str));

    # lowerChars/upperChars - character lists
    lowerChars = strings.stringToCharacters "abcdefghijklmnopqrstuvwxyz";
    upperChars = strings.stringToCharacters "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    # fileContents :: Path -> String
    fileContents = path: strings.trim (P.read-file path);

    # escapeShellArgs :: [String] -> String
    escapeShellArgs = args: P.join " " (P.map P.escape-shell args);

    # isPath :: a -> Bool
    isPath = P.is-path;

    # isStorePath :: String -> Bool
    isStorePath = str: P.starts-with builtins.storeDir (toString str);

    # isStringLike :: a -> Bool
    isStringLike = x: P.is-string x || (P.is-attrs x && x ? __toString);

    # toInt :: String -> Int
    toInt =
      str:
      let
        parsed = builtins.fromJSON str;
      in
      if P.is-int parsed then parsed else throw "not an integer: ${str}";

    # getName :: Derivation -> String
    getName = drv: drv.pname or (builtins.parseDrvName drv.name).name;

    # getVersion :: Derivation -> String
    getVersion = drv: drv.version or (builtins.parseDrvName drv.name).version;

    # versionOlder :: String -> String -> Bool
    versionOlder = a: b: builtins.compareVersions a b < 0;

    # versionAtLeast :: String -> String -> Bool
    versionAtLeast = a: b: builtins.compareVersions a b >= 0;

    # nameFromURL :: String -> String -> String
    nameFromURL =
      url: sep:
      let
        parts = P.split "/" url;
      in
      P.last (P.split sep (P.last parts));

    # enableFeature :: Bool -> String -> String
    enableFeature = enable: name: "--${if enable then "enable" else "disable"}-${name}";

    # withFeature :: Bool -> String -> String
    withFeature = with_: name: "--${if with_ then "with" else "without"}-${name}";

    # fixedWidthString :: Int -> String -> String -> String
    fixedWidthString =
      width: filler: str:
      let
        len = P.string-length str;
        pad = width - len;
      in
      if pad <= 0 then str else P.join "" (P.replicate pad filler) + str;

    # fixedWidthNumber :: Int -> Int -> String
    fixedWidthNumber = width: n: strings.fixedWidthString width "0" (toString n);

    # escapeRegex :: String -> String
    escapeRegex = P.escape [
      "\\"
      "^"
      "$"
      "."
      "|"
      "?"
      "*"
      "+"
      "("
      ")"
      "["
      "]"
      "{"
      "}"
    ];

    # escapeXML :: String -> String
    escapeXML = P.replace [ "&" "<" ">" "'" "\"" ] [ "&amp;" "&lt;" "&gt;" "&apos;" "&quot;" ];

    # match :: String -> String -> [String]?
    inherit (builtins) match;

    # split :: String -> String -> [String]
    inherit (builtins) split;

    # cmake helpers
    cmakeBool = val: if val then "ON" else "OFF";
    cmakeFeature = name: val: "-D${name}=${val}";
    cmakeOptionType =
      type: name: val:
      "-D${name}:${type}=${val}";

    # meson helpers
    mesonBool = val: if val then "true" else "false";
    mesonEnable = val: if val then "enabled" else "disabled";
    mesonOption = name: val: "-D${name}=${val}";

    # makeSearchPath :: String -> [Path] -> String
    makeSearchPath = subDir: paths: P.join ":" (P.map (path: "${path}/${subDir}") paths);

    makeLibraryPath = strings.makeSearchPath "lib";
    makeBinPath = strings.makeSearchPath "bin";
    makeIncludePath = strings.makeSearchPath "include";

    # Shell variable helpers
    toShellVar = name: value: "${name}=${P.escape-shell (toString value)}";
    toShellVars = attrs: P.join "\n" (P.map-to-list strings.toShellVar attrs);

    # join - alias for concatStringsSep (Prelude has this as P.join)
    inherit (P) join;

    # addContextFrom :: String -> String -> String
    addContextFrom = a: b: builtins.substring 0 0 a + b;

    # splitStringBy: Not implemented - use builtins.split with regex instead
    # splitStringBy = pred: str: ...;

    # toCamelCase :: String -> String
    toCamelCase =
      str:
      let
        parts = P.split "-" str;
        cap = s: P.to-upper (P.substring 0 1 s) + P.substring 1 (P.string-length s) s;
      in
      P.head parts + P.join "" (P.map cap (P.tail parts));

    # toSentenceCase :: String -> String
    toSentenceCase =
      str: P.to-upper (P.substring 0 1 str) + P.to-lower (P.substring 1 (P.string-length str) str);

    # trimWith :: { start?: Bool, end?: Bool } -> String -> String
    trimWith =
      {
        start ? true,
        end ? true,
      }:
      str:
      let
        trim-start =
          s:
          let
            m = builtins.match "[ \t\n]*(.*)" s;
          in
          if m == null then s else P.head m;
        trim-end =
          s:
          let
            m = builtins.match "(.*[^ \t\n])[ \t\n]*" s;
          in
          if m == null then s else P.head m;
      in
      (if end then trim-end else P.id) ((if start then trim-start else P.id) str);

    # replaceString - alias for single replacement
    replaceString = old: new: P.replace [ old ] [ new ];

    # New additions - Prelude has these, legacy doesn't!
    padLeft = P.pad-left;
    padRight = P.pad-right;
    inherit (P) capitalize;
    inherit (P) uncapitalize;
    toKebab = P.to-kebab;
    isEmpty = P.is-empty;
    isBlank = P.is-blank;
    inherit (P) repeat;
  };

  # ════════════════════════════════════════════════════════════════════════════
  # FIXED POINTS
  # ════════════════════════════════════════════════════════════════════════════
  fixedPoints = {
    inherit (P) fix;
    fix' =
      f:
      let
        x = f x // {
          __unfix__ = f;
        };
      in
      x;

    converge =
      f: x:
      let
        x' = f x;
      in
      if x == x' then x else fixedPoints.converge f x';

    extends =
      f: rattrs: self:
      let
        super = rattrs self;
      in
      super // f self super;

    composeExtensions =
      f: g: final: prev:
      let
        f-applied = f final prev;
        prev' = prev // f-applied;
      in
      f-applied // g final prev';

    composeManyExtensions = P.fold fixedPoints.composeExtensions (_: _: { });

    makeExtensible =
      rattrs:
      fixedPoints.fix' (
        self:
        rattrs self
        // {
          extend = f: fixedPoints.makeExtensible (fixedPoints.extends f rattrs);
        }
      );

    makeExtensibleWithCustomName =
      name: rattrs:
      fixedPoints.fix' (
        self:
        rattrs self
        // {
          ${name} = f: fixedPoints.makeExtensibleWithCustomName name (fixedPoints.extends f rattrs);
        }
      );

    toExtension =
      f: final: _prev:
      f final;
  };

  # ════════════════════════════════════════════════════════════════════════════
  # DEBUG
  # ════════════════════════════════════════════════════════════════════════════
  debug = {
    inherit (P) trace;
    traceVal = P.trace-val;
    traceSeq = x: y: builtins.trace (builtins.deepSeq x x) y;

    traceIf = cond: msg: if cond then builtins.trace msg else P.id;
    traceValFn = f: x: builtins.trace (f x) x;
    traceSeqN = _depth: x: builtins.trace (builtins.deepSeq x x) x;

    traceValSeq = x: builtins.trace (builtins.deepSeq x x) x;
    traceValSeqFn = f: x: builtins.trace (builtins.deepSeq (f x) (f x)) x;
    traceValSeqN = _depth: debug.traceValSeq;
    traceValSeqNFn = _depth: debug.traceValSeqFn;
    traceFnSeqN =
      _depth: f: x:
      builtins.trace (builtins.deepSeq (f x) (f x)) x;

    addErrorContext = builtins.addErrorContext or (_context: x: x);
    unsafeGetAttrPos = builtins.unsafeGetAttrPos or (_name: _attrs: null);

    runTests =
      tests:
      P.map-to-list (
        name: test:
        if test.expr == test.expected then
          {
            inherit name;
            success = true;
          }
        else
          {
            inherit name;
            success = false;
            inherit (test) expected;
            result = test.expr;
          }
      ) tests;

    testAllTrue = tests: P.all (t: t.success) (debug.runTests tests);
  };

  # ════════════════════════════════════════════════════════════════════════════
  # ASSERTS
  # ════════════════════════════════════════════════════════════════════════════
  asserts = {
    assertMsg = P.assert-msg;
    assertOneOf =
      name: val: xs:
      P.assert-msg (P.elem val xs) "${name} must be one of ${builtins.toJSON xs}, but got: ${builtins.toJSON val}";
  };

  # ════════════════════════════════════════════════════════════════════════════
  # THE UNIFIED LIB
  # ════════════════════════════════════════════════════════════════════════════
  lib =
    trivial
    // lists
    // attrsets
    // strings
    // fixedPoints
    // debug
    // asserts
    // {
      inherit
        trivial
        lists
        attrsets
        strings
        fixedPoints
        debug
        asserts
        ;

      # Module system would be imported from nixpkgs
      # modules = ...;
      # options = ...;
      # types = ...;

      # Aliases for common patterns
      inherit (lists) optional;
      inherit (lists) optionals;
      inherit (strings) optionalString;
      inherit (attrsets) optionalAttrs;
      mkIf = cond: val: if cond then val else { };
      mkMerge = P.merge-all;

      # ╔════════════════════════════════════════════════════════════════════════╗
      # ║  WARNING: mkForce/mkDefault are STUBS that DO NOT work with NixOS      ║
      # ║  module system priorities! They create attribute sets that look like   ║
      # ║  priority markers but won't be processed by the module system.         ║
      # ║                                                                        ║
      # ║  For NixOS modules, use the real lib.mkForce/lib.mkDefault from        ║
      # ║  nixpkgs. These stubs exist only for basic non-module compatibility.   ║
      # ╚════════════════════════════════════════════════════════════════════════╝
      mkForce =
        builtins.trace
          "WARNING: lib.mkForce from aleph prelude is a stub - use nixpkgs lib for module system"
          (val: {
            _type = "override";
            priority = 50;
            content = val;
          });
      mkDefault =
        builtins.trace
          "WARNING: lib.mkDefault from aleph prelude is a stub - use nixpkgs lib for module system"
          (val: {
            _type = "override";
            priority = 1000;
            content = val;
          });

      # The prelude itself for direct access
      inherit prelude;
    };

in
lib
