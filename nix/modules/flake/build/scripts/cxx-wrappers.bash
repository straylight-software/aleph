cp @scriptsDir@/cxx-wrapper.bash bin/cxx
chmod +x bin/cxx

# compile_commands.json generator for clangd/clang-tidy
cp @scriptsDir@/compdb.bash bin/compdb
chmod +x bin/compdb
