/*
 * ld-shim.c - Fake linker that captures link invocations
 *
 * This shim replaces ld/ld.lld/ld.gold during build analysis.
 * Instead of linking, it:
 *   1. Records the full invocation to a log file
 *   2. Reads metadata from input .o files (written by cc-shim)
 *   3. Emits a fake executable/library with combined metadata
 *
 * The fake output contains:
 *   - Symbol: __armitage_objects = "obj1.o:obj2.o:..."
 *   - Symbol: __armitage_libs = "z:pthread:ssl:..."
 *   - Symbol: __armitage_libpaths = "/path1:/path2:..."
 *   - Symbol: __armitage_rpaths = "/rpath1:/rpath2:..."
 *   - Symbol: __armitage_output = "output"
 *   - Plus all __armitage_* symbols from input .o files
 *
 * Build: clang -o ld-shim ld-shim.c
 * Usage: LD=/path/to/ld-shim cmake ..
 */

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ELF structures */
#define EI_NIDENT 16
#define ET_EXEC 2
#define ET_DYN 3
#define ET_REL 1
#define EM_X86_64 62
#define SHT_SYMTAB 2
#define SHT_STRTAB 3
#define STB_GLOBAL 1
#define STT_OBJECT 1

typedef struct {
  unsigned char e_ident[EI_NIDENT];
  uint16_t e_type;
  uint16_t e_machine;
  uint32_t e_version;
  uint64_t e_entry;
  uint64_t e_phoff;
  uint64_t e_shoff;
  uint32_t e_flags;
  uint16_t e_ehsize;
  uint16_t e_phentsize;
  uint16_t e_phnum;
  uint16_t e_shentsize;
  uint16_t e_shnum;
  uint16_t e_shstrndx;
} Elf64_Ehdr;

typedef struct {
  uint32_t sh_name;
  uint32_t sh_type;
  uint64_t sh_flags;
  uint64_t sh_addr;
  uint64_t sh_offset;
  uint64_t sh_size;
  uint32_t sh_link;
  uint32_t sh_info;
  uint64_t sh_addralign;
  uint64_t sh_entsize;
} Elf64_Shdr;

typedef struct {
  uint32_t st_name;
  unsigned char st_info;
  unsigned char st_other;
  uint16_t st_shndx;
  uint64_t st_value;
  uint64_t st_size;
} Elf64_Sym;

#define ELF64_ST_INFO(bind, type) (((bind) << 4) + ((type) & 0xf))

/* Max sizes */
#define MAX_OBJECTS 1024
#define MAX_LIBS 256
#define MAX_LIBPATHS 256
#define MAX_RPATHS 256
#define MAX_PATH 4096
#define MAX_METADATA 262144 /* 256KB for combined metadata */

/* Parsed arguments */
static char *objects[MAX_OBJECTS];
static int num_objects = 0;
static char *libs[MAX_LIBS];
static int num_libs = 0;
static char *libpaths[MAX_LIBPATHS];
static int num_libpaths = 0;
static char *rpaths[MAX_RPATHS];
static int num_rpaths = 0;
static char *output_file = NULL;
static int shared = 0;
static int pie = 0;

/* Collected metadata from input objects */
static char all_sources[MAX_METADATA] = "";
static char all_includes[MAX_METADATA] = "";
static char all_defines[MAX_METADATA] = "";
static char all_flags[MAX_METADATA] = "";

/* Log file path from environment */
static const char *get_log_path(void) {
  const char *path = getenv("ARMITAGE_SHIM_LOG");
  return path ? path : "/tmp/armitage-shim.log";
}

/* Append to log file */
static void log_invocation(int argc, char **argv) {
  const char *log_path = get_log_path();
  FILE *f = fopen(log_path, "a");
  if (!f)
    return;

  time_t now = time(NULL);
  struct tm *tm = localtime(&now);
  fprintf(f, "[%04d-%02d-%02d %02d:%02d:%02d] ", tm->tm_year + 1900,
          tm->tm_mon + 1, tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec);

  fprintf(f, "pid=%d LD ", getpid());
  for (int i = 0; i < argc; i++) {
    fprintf(f, "%s%s", i ? " " : "", argv[i]);
  }
  fprintf(f, "\n");
  fclose(f);
}

/* Check if file is an object file */
static int is_object_file(const char *path) {
  const char *ext = strrchr(path, '.');
  if (!ext)
    return 0;
  return strcmp(ext, ".o") == 0 || strcmp(ext, ".a") == 0;
}

/* Parse command line arguments */
static void parse_args(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    const char *arg = argv[i];

    if (strcmp(arg, "-o") == 0 && i + 1 < argc) {
      output_file = argv[++i];
    } else if (strncmp(arg, "-o", 2) == 0 && arg[2]) {
      output_file = (char *)(arg + 2);
    } else if (strcmp(arg, "-L") == 0 && i + 1 < argc) {
      if (num_libpaths < MAX_LIBPATHS)
        libpaths[num_libpaths++] = argv[++i];
    } else if (strncmp(arg, "-L", 2) == 0) {
      if (num_libpaths < MAX_LIBPATHS)
        libpaths[num_libpaths++] = (char *)(arg + 2);
    } else if (strcmp(arg, "-l") == 0 && i + 1 < argc) {
      if (num_libs < MAX_LIBS)
        libs[num_libs++] = argv[++i];
    } else if (strncmp(arg, "-l", 2) == 0) {
      if (num_libs < MAX_LIBS)
        libs[num_libs++] = (char *)(arg + 2);
    } else if (strcmp(arg, "-rpath") == 0 && i + 1 < argc) {
      if (num_rpaths < MAX_RPATHS)
        rpaths[num_rpaths++] = argv[++i];
    } else if (strncmp(arg, "-rpath=", 7) == 0) {
      if (num_rpaths < MAX_RPATHS)
        rpaths[num_rpaths++] = (char *)(arg + 7);
    } else if (strcmp(arg, "-shared") == 0) {
      shared = 1;
    } else if (strcmp(arg, "-pie") == 0) {
      pie = 1;
    } else if (strcmp(arg, "--eh-frame-hdr") == 0 ||
               strcmp(arg, "--build-id") == 0 ||
               strcmp(arg, "--hash-style=gnu") == 0 ||
               strcmp(arg, "--as-needed") == 0 ||
               strcmp(arg, "--no-as-needed") == 0 || strcmp(arg, "-z") == 0 ||
               strncmp(arg, "-m", 2) == 0 ||
               strncmp(arg, "--sysroot", 9) == 0) {
      /* Skip known flags */
      if (strcmp(arg, "-z") == 0 && i + 1 < argc)
        i++;
    } else if (arg[0] != '-' && is_object_file(arg)) {
      if (num_objects < MAX_OBJECTS)
        objects[num_objects++] = (char *)arg;
    }
  }
}

/* Append string to buffer with separator */
static void append_with_sep(char *buf, size_t bufsize, const char *str,
                            char sep) {
  size_t len = strlen(buf);
  if (len > 0 && len < bufsize - 1) {
    buf[len++] = sep;
    buf[len] = '\0';
  }
  size_t slen = strlen(str);
  if (len + slen < bufsize - 1) {
    memcpy(buf + len, str, slen + 1);
  }
}

/* Read metadata from an object file */
static void read_object_metadata(const char *path) {
  int fd = open(path, O_RDONLY);
  if (fd < 0)
    return;

  struct stat st;
  if (fstat(fd, &st) < 0) {
    close(fd);
    return;
  }

  if (st.st_size < (off_t)sizeof(Elf64_Ehdr)) {
    close(fd);
    return;
  }

  void *map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED)
    return;

  Elf64_Ehdr *ehdr = (Elf64_Ehdr *)map;

  /* Verify ELF magic */
  if (ehdr->e_ident[0] != 0x7f || ehdr->e_ident[1] != 'E' ||
      ehdr->e_ident[2] != 'L' || ehdr->e_ident[3] != 'F') {
    munmap(map, st.st_size);
    return;
  }

  /* Find .symtab and .strtab */
  Elf64_Shdr *shdrs = (Elf64_Shdr *)((char *)map + ehdr->e_shoff);
  Elf64_Shdr *symtab = NULL;
  Elf64_Shdr *strtab = NULL;
  char *data = NULL;

  for (int i = 0; i < ehdr->e_shnum; i++) {
    if (shdrs[i].sh_type == SHT_SYMTAB) {
      symtab = &shdrs[i];
      if (symtab->sh_link < ehdr->e_shnum) {
        strtab = &shdrs[symtab->sh_link];
      }
    }
    /* Find .data section for symbol values */
    if (shdrs[i].sh_type == 1 && shdrs[i].sh_flags == 3) {
      data = (char *)map + shdrs[i].sh_offset;
    }
  }

  if (!symtab || !strtab || !data) {
    munmap(map, st.st_size);
    return;
  }

  char *strings = (char *)map + strtab->sh_offset;
  Elf64_Sym *syms = (Elf64_Sym *)((char *)map + symtab->sh_offset);
  int num_syms = symtab->sh_size / sizeof(Elf64_Sym);

  for (int i = 0; i < num_syms; i++) {
    const char *name = strings + syms[i].st_name;
    if (strncmp(name, "__armitage_", 11) != 0)
      continue;

    const char *value = data + syms[i].st_value;
    if (!value[0])
      continue; /* Skip empty values */

    if (strcmp(name, "__armitage_sources") == 0) {
      append_with_sep(all_sources, sizeof(all_sources), value, ':');
    } else if (strcmp(name, "__armitage_includes") == 0) {
      append_with_sep(all_includes, sizeof(all_includes), value, ':');
    } else if (strcmp(name, "__armitage_defines") == 0) {
      append_with_sep(all_defines, sizeof(all_defines), value, ':');
    } else if (strcmp(name, "__armitage_flags") == 0) {
      append_with_sep(all_flags, sizeof(all_flags), value, ':');
    }
  }

  munmap(map, st.st_size);
}

/* Join strings with separator */
static int join_strings(char *buf, size_t bufsize, char **strs, int count,
                        char sep) {
  size_t pos = 0;
  for (int i = 0; i < count && pos < bufsize - 1; i++) {
    if (i > 0 && pos < bufsize - 1)
      buf[pos++] = sep;
    size_t len = strlen(strs[i]);
    if (pos + len >= bufsize - 1)
      break;
    memcpy(buf + pos, strs[i], len);
    pos += len;
  }
  buf[pos] = '\0';
  return pos;
}

/* Write a fake executable with combined metadata */
static int write_fake_executable(const char *path) {
  /* Build metadata strings */
  char objects_str[MAX_METADATA] = "";
  char libs_str[MAX_METADATA] = "";
  char libpaths_str[MAX_METADATA] = "";
  char rpaths_str[MAX_METADATA] = "";

  join_strings(objects_str, sizeof(objects_str), objects, num_objects, ':');
  join_strings(libs_str, sizeof(libs_str), libs, num_libs, ':');
  join_strings(libpaths_str, sizeof(libpaths_str), libpaths, num_libpaths, ':');
  join_strings(rpaths_str, sizeof(rpaths_str), rpaths, num_rpaths, ':');

  /* Symbol names - link info plus aggregated compile info */
  const char *sym_names[] = {
      "", /* null symbol */
      "__armitage_objects",
      "__armitage_libs",
      "__armitage_libpaths",
      "__armitage_rpaths",
      "__armitage_output",
      "__armitage_all_sources",
      "__armitage_all_includes",
      "__armitage_all_defines",
      "__armitage_all_flags",
  };
  const char *sym_values[] = {
      "",           objects_str,  libs_str,
      libpaths_str, rpaths_str,   output_file ? output_file : "",
      all_sources,  all_includes, all_defines,
      all_flags,
  };
  const int num_syms = 10;

  /* Build string table */
  char strtab[MAX_METADATA];
  size_t strtab_size = 1;
  strtab[0] = '\0';

  uint32_t sym_name_offsets[10];
  sym_name_offsets[0] = 0;

  for (int i = 1; i < num_syms; i++) {
    sym_name_offsets[i] = strtab_size;
    size_t len = strlen(sym_names[i]) + 1;
    memcpy(strtab + strtab_size, sym_names[i], len);
    strtab_size += len;
  }

  /* Build data section */
  char data[MAX_METADATA];
  size_t data_size = 0;
  uint64_t sym_offsets[10];
  uint64_t sym_sizes[10];

  sym_offsets[0] = 0;
  sym_sizes[0] = 0;

  for (int i = 1; i < num_syms; i++) {
    sym_offsets[i] = data_size;
    size_t len = strlen(sym_values[i]) + 1;
    sym_sizes[i] = len;
    if (data_size + len < sizeof(data)) {
      memcpy(data + data_size, sym_values[i], len);
      data_size += len;
    }
  }

  /* Section header string table */
  const char shstrtab[] = "\0.shstrtab\0.strtab\0.symtab\0.data";
  const size_t shstrtab_size = sizeof(shstrtab);
  const uint32_t sh_shstrtab = 1;
  const uint32_t sh_strtab = 11;
  const uint32_t sh_symtab = 19;
  const uint32_t sh_data = 27;

  /* Calculate offsets */
  size_t ehdr_size = sizeof(Elf64_Ehdr);
  size_t shdr_size = sizeof(Elf64_Shdr);

  size_t data_off = ehdr_size;
  size_t shstrtab_off = data_off + data_size;
  size_t strtab_off = shstrtab_off + shstrtab_size;
  size_t symtab_off = strtab_off + strtab_size;
  size_t symtab_size = num_syms * sizeof(Elf64_Sym);
  size_t shdr_off = symtab_off + symtab_size;
  shdr_off = (shdr_off + 7) & ~7;

  /* Build ELF header */
  Elf64_Ehdr ehdr = {0};
  ehdr.e_ident[0] = 0x7f;
  ehdr.e_ident[1] = 'E';
  ehdr.e_ident[2] = 'L';
  ehdr.e_ident[3] = 'F';
  ehdr.e_ident[4] = 2;
  ehdr.e_ident[5] = 1;
  ehdr.e_ident[6] = 1;
  ehdr.e_type = shared ? ET_DYN : (pie ? ET_DYN : ET_EXEC);
  ehdr.e_machine = EM_X86_64;
  ehdr.e_version = 1;
  ehdr.e_shoff = shdr_off;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = 5;
  ehdr.e_shstrndx = 1;

  /* Build section headers */
  Elf64_Shdr shdrs[5] = {0};

  shdrs[1].sh_name = sh_shstrtab;
  shdrs[1].sh_type = SHT_STRTAB;
  shdrs[1].sh_offset = shstrtab_off;
  shdrs[1].sh_size = shstrtab_size;
  shdrs[1].sh_addralign = 1;

  shdrs[2].sh_name = sh_strtab;
  shdrs[2].sh_type = SHT_STRTAB;
  shdrs[2].sh_offset = strtab_off;
  shdrs[2].sh_size = strtab_size;
  shdrs[2].sh_addralign = 1;

  shdrs[3].sh_name = sh_symtab;
  shdrs[3].sh_type = SHT_SYMTAB;
  shdrs[3].sh_offset = symtab_off;
  shdrs[3].sh_size = symtab_size;
  shdrs[3].sh_link = 2;
  shdrs[3].sh_info = 1;
  shdrs[3].sh_addralign = 8;
  shdrs[3].sh_entsize = sizeof(Elf64_Sym);

  shdrs[4].sh_name = sh_data;
  shdrs[4].sh_type = 1;
  shdrs[4].sh_flags = 3;
  shdrs[4].sh_offset = data_off;
  shdrs[4].sh_size = data_size;
  shdrs[4].sh_addralign = 1;

  /* Build symbol table */
  Elf64_Sym syms[10] = {0};
  for (int i = 1; i < num_syms; i++) {
    syms[i].st_name = sym_name_offsets[i];
    syms[i].st_info = ELF64_ST_INFO(STB_GLOBAL, STT_OBJECT);
    syms[i].st_shndx = 4;
    syms[i].st_value = sym_offsets[i];
    syms[i].st_size = sym_sizes[i];
  }

  /* Write the file */
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0755);
  if (fd < 0)
    return -1;

  write(fd, &ehdr, sizeof(ehdr));
  write(fd, data, data_size);
  write(fd, shstrtab, shstrtab_size);
  write(fd, strtab, strtab_size);
  write(fd, syms, symtab_size);

  size_t current = symtab_off + symtab_size;
  while (current < shdr_off) {
    char zero = 0;
    write(fd, &zero, 1);
    current++;
  }

  write(fd, shdrs, sizeof(shdrs));
  close(fd);

  return 0;
}

int main(int argc, char **argv) {
  log_invocation(argc, argv);
  parse_args(argc, argv);

  /* Read metadata from all input objects */
  for (int i = 0; i < num_objects; i++) {
    read_object_metadata(objects[i]);
  }

  /* Emit fake executable */
  const char *out = output_file ? output_file : "a.out";
  if (write_fake_executable(out) < 0) {
    fprintf(stderr, "ld-shim: failed to write %s\n", out);
    return 1;
  }

  return 0;
}
