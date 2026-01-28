/*
 * ar-shim.c - Fake archiver that captures archive operations
 *
 * This shim replaces ar/llvm-ar during build analysis.
 * Instead of creating archives, it:
 *   1. Records the full invocation to a log file
 *   2. Reads metadata from input .o files (written by cc-shim)
 *   3. Emits a fake .a file with combined metadata
 *
 * The fake .a is actually an ELF .o file containing:
 *   - Symbol: __armitage_archive = "libfoo.a"
 *   - Symbol: __armitage_members = "obj1.o:obj2.o:..."
 *   - Plus all __armitage_* symbols from member objects
 *
 * Build: clang -o ar-shim ar-shim.c
 * Usage: AR=/path/to/ar-shim cmake ..
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
#define MAX_MEMBERS 1024
#define MAX_PATH 4096
#define MAX_METADATA 262144

/* Parsed arguments */
static char *members[MAX_MEMBERS];
static int num_members = 0;
static char *archive_name = NULL;
static char operation = 0; /* r, q, t, x, etc */

/* Collected metadata */
static char all_sources[MAX_METADATA] = "";
static char all_includes[MAX_METADATA] = "";
static char all_defines[MAX_METADATA] = "";
static char all_flags[MAX_METADATA] = "";

/* Log file path */
static const char *get_log_path(void) {
  const char *path = getenv("ARMITAGE_SHIM_LOG");
  return path ? path : "/tmp/armitage-shim.log";
}

static void log_invocation(int argc, char **argv) {
  const char *log_path = get_log_path();
  FILE *f = fopen(log_path, "a");
  if (!f)
    return;

  time_t now = time(NULL);
  struct tm *tm = localtime(&now);
  fprintf(f, "[%04d-%02d-%02d %02d:%02d:%02d] ", tm->tm_year + 1900,
          tm->tm_mon + 1, tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec);

  fprintf(f, "pid=%d AR ", getpid());
  for (int i = 0; i < argc; i++) {
    fprintf(f, "%s%s", i ? " " : "", argv[i]);
  }
  fprintf(f, "\n");
  fclose(f);
}

static int is_object_file(const char *path) {
  const char *ext = strrchr(path, '.');
  if (!ext)
    return 0;
  return strcmp(ext, ".o") == 0;
}

/* Parse ar-style arguments: ar rcs libfoo.a obj1.o obj2.o */
static void parse_args(int argc, char **argv) {
  int i = 1;

  /* First arg might be operation flags (rcs, etc) */
  if (i < argc && argv[i][0] != '-' && !is_object_file(argv[i]) &&
      strstr(argv[i], ".a") == NULL) {
    /* This is the operation string like "rcs" */
    operation = argv[i][0];
    i++;
  }

  /* Next should be archive name */
  if (i < argc) {
    const char *ext = strrchr(argv[i], '.');
    if (ext && strcmp(ext, ".a") == 0) {
      archive_name = argv[i];
      i++;
    }
  }

  /* Rest are member objects */
  for (; i < argc; i++) {
    if (is_object_file(argv[i])) {
      if (num_members < MAX_MEMBERS)
        members[num_members++] = argv[i];
    } else if (strstr(argv[i], ".a") != NULL && !archive_name) {
      archive_name = argv[i];
    }
  }
}

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

  if (ehdr->e_ident[0] != 0x7f || ehdr->e_ident[1] != 'E' ||
      ehdr->e_ident[2] != 'L' || ehdr->e_ident[3] != 'F') {
    munmap(map, st.st_size);
    return;
  }

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
      continue;

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

/* Write fake archive (actually an ELF .o with metadata) */
static int write_fake_archive(const char *path) {
  char members_str[MAX_METADATA] = "";
  join_strings(members_str, sizeof(members_str), members, num_members, ':');

  const char *sym_names[] = {
      "",
      "__armitage_archive",
      "__armitage_members",
      "__armitage_all_sources",
      "__armitage_all_includes",
      "__armitage_all_defines",
      "__armitage_all_flags",
  };
  const char *sym_values[] = {
      "",           archive_name ? archive_name : "",
      members_str,  all_sources,
      all_includes, all_defines,
      all_flags,
  };
  const int num_syms = 7;

  /* Build string table */
  char strtab[MAX_METADATA];
  size_t strtab_size = 1;
  strtab[0] = '\0';

  uint32_t sym_name_offsets[7];
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
  uint64_t sym_offsets[7];
  uint64_t sym_sizes[7];

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

  const char shstrtab[] = "\0.shstrtab\0.strtab\0.symtab\0.data";
  const size_t shstrtab_size = sizeof(shstrtab);
  const uint32_t sh_shstrtab = 1;
  const uint32_t sh_strtab = 11;
  const uint32_t sh_symtab = 19;
  const uint32_t sh_data = 27;

  size_t data_off = sizeof(Elf64_Ehdr);
  size_t shstrtab_off = data_off + data_size;
  size_t strtab_off = shstrtab_off + shstrtab_size;
  size_t symtab_off = strtab_off + strtab_size;
  size_t symtab_size = num_syms * sizeof(Elf64_Sym);
  size_t shdr_off = symtab_off + symtab_size;
  shdr_off = (shdr_off + 7) & ~7;

  Elf64_Ehdr ehdr = {0};
  ehdr.e_ident[0] = 0x7f;
  ehdr.e_ident[1] = 'E';
  ehdr.e_ident[2] = 'L';
  ehdr.e_ident[3] = 'F';
  ehdr.e_ident[4] = 2;
  ehdr.e_ident[5] = 1;
  ehdr.e_ident[6] = 1;
  ehdr.e_type = ET_REL;
  ehdr.e_machine = EM_X86_64;
  ehdr.e_version = 1;
  ehdr.e_shoff = shdr_off;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = 5;
  ehdr.e_shstrndx = 1;

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

  Elf64_Sym syms[7] = {0};
  for (int i = 1; i < num_syms; i++) {
    syms[i].st_name = sym_name_offsets[i];
    syms[i].st_info = ELF64_ST_INFO(STB_GLOBAL, STT_OBJECT);
    syms[i].st_shndx = 4;
    syms[i].st_value = sym_offsets[i];
    syms[i].st_size = sym_sizes[i];
  }

  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
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

  /* For 't' (table) operation, just succeed */
  if (operation == 't') {
    return 0;
  }

  /* Read metadata from all input objects */
  for (int i = 0; i < num_members; i++) {
    read_object_metadata(members[i]);
  }

  /* Emit fake archive */
  if (archive_name) {
    if (write_fake_archive(archive_name) < 0) {
      fprintf(stderr, "ar-shim: failed to write %s\n", archive_name);
      return 1;
    }
  }

  return 0;
}
