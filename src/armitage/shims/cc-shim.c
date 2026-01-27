/*
 * cc-shim.c - Fake compiler that captures invocations
 *
 * This shim replaces cc/c++/gcc/g++/clang/clang++ during build analysis.
 * Instead of compiling, it:
 *   1. Records the full invocation to a log file
 *   2. Emits a fake .o file with metadata encoded in ELF symbols
 *
 * The fake .o contains:
 *   - Symbol: __armitage_sources = "src1.c:src2.c:..."
 *   - Symbol: __armitage_includes = "/path1:/path2:..."
 *   - Symbol: __armitage_defines = "FOO=1:BAR:..."
 *   - Symbol: __armitage_flags = "-O2:-Wall:..."
 *   - Symbol: __armitage_output = "output.o"
 *
 * Build: clang -o cc-shim cc-shim.c
 * Usage: CC=/path/to/cc-shim cmake ..
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <stdint.h>

/* ELF structures - minimal, 64-bit only */
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
#define MAX_SOURCES 256
#define MAX_INCLUDES 256
#define MAX_DEFINES 256
#define MAX_FLAGS 256
#define MAX_PATH 4096
#define MAX_METADATA 65536

/* Parsed arguments */
static char *sources[MAX_SOURCES];
static int num_sources = 0;
static char *includes[MAX_INCLUDES];
static int num_includes = 0;
static char *defines[MAX_DEFINES];
static int num_defines = 0;
static char *flags[MAX_FLAGS];
static int num_flags = 0;
static char *output_file = NULL;
static int compile_only = 0;  /* -c flag */
static int preprocess_only = 0;  /* -E flag */
static int syntax_only = 0;  /* -fsyntax-only */

/* Log file path from environment */
static const char *get_log_path(void) {
    const char *path = getenv("ARMITAGE_SHIM_LOG");
    return path ? path : "/tmp/armitage-shim.log";
}

/* Append to log file */
static void log_invocation(int argc, char **argv) {
    const char *log_path = get_log_path();
    FILE *f = fopen(log_path, "a");
    if (!f) return;
    
    /* Timestamp */
    time_t now = time(NULL);
    struct tm *tm = localtime(&now);
    fprintf(f, "[%04d-%02d-%02d %02d:%02d:%02d] ",
            tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
            tm->tm_hour, tm->tm_min, tm->tm_sec);
    
    /* PID and command */
    fprintf(f, "pid=%d ", getpid());
    for (int i = 0; i < argc; i++) {
        fprintf(f, "%s%s", i ? " " : "", argv[i]);
    }
    fprintf(f, "\n");
    fclose(f);
}

/* Check if file looks like a source file */
static int is_source_file(const char *path) {
    const char *ext = strrchr(path, '.');
    if (!ext) return 0;
    return strcmp(ext, ".c") == 0 ||
           strcmp(ext, ".cc") == 0 ||
           strcmp(ext, ".cpp") == 0 ||
           strcmp(ext, ".cxx") == 0 ||
           strcmp(ext, ".C") == 0 ||
           strcmp(ext, ".c++") == 0 ||
           strcmp(ext, ".cu") == 0 ||
           strcmp(ext, ".m") == 0 ||
           strcmp(ext, ".mm") == 0 ||
           strcmp(ext, ".S") == 0 ||
           strcmp(ext, ".s") == 0;
}

/* Parse command line arguments */
static void parse_args(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        
        if (strcmp(arg, "-c") == 0) {
            compile_only = 1;
        } else if (strcmp(arg, "-E") == 0) {
            preprocess_only = 1;
        } else if (strcmp(arg, "-fsyntax-only") == 0) {
            syntax_only = 1;
        } else if (strcmp(arg, "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strncmp(arg, "-o", 2) == 0 && arg[2]) {
            output_file = (char *)(arg + 2);
        } else if (strcmp(arg, "-I") == 0 && i + 1 < argc) {
            if (num_includes < MAX_INCLUDES)
                includes[num_includes++] = argv[++i];
        } else if (strncmp(arg, "-I", 2) == 0) {
            if (num_includes < MAX_INCLUDES)
                includes[num_includes++] = (char *)(arg + 2);
        } else if (strcmp(arg, "-isystem") == 0 && i + 1 < argc) {
            if (num_includes < MAX_INCLUDES)
                includes[num_includes++] = argv[++i];
        } else if (strcmp(arg, "-D") == 0 && i + 1 < argc) {
            if (num_defines < MAX_DEFINES)
                defines[num_defines++] = argv[++i];
        } else if (strncmp(arg, "-D", 2) == 0) {
            if (num_defines < MAX_DEFINES)
                defines[num_defines++] = (char *)(arg + 2);
        } else if (arg[0] == '-') {
            /* Other flag */
            if (num_flags < MAX_FLAGS)
                flags[num_flags++] = (char *)arg;
        } else if (is_source_file(arg)) {
            if (num_sources < MAX_SOURCES)
                sources[num_sources++] = (char *)arg;
        }
        /* Skip other positional args (response files, etc) */
    }
}

/* Join strings with separator */
static int join_strings(char *buf, size_t bufsize, char **strs, int count, char sep) {
    size_t pos = 0;
    for (int i = 0; i < count && pos < bufsize - 1; i++) {
        if (i > 0 && pos < bufsize - 1) buf[pos++] = sep;
        size_t len = strlen(strs[i]);
        if (pos + len >= bufsize - 1) break;
        memcpy(buf + pos, strs[i], len);
        pos += len;
    }
    buf[pos] = '\0';
    return pos;
}

/* Write a minimal ELF .o file with metadata in symbols */
static int write_fake_object(const char *path) {
    /* Build metadata strings */
    char sources_str[MAX_METADATA] = "";
    char includes_str[MAX_METADATA] = "";
    char defines_str[MAX_METADATA] = "";
    char flags_str[MAX_METADATA] = "";
    
    join_strings(sources_str, sizeof(sources_str), sources, num_sources, ':');
    join_strings(includes_str, sizeof(includes_str), includes, num_includes, ':');
    join_strings(defines_str, sizeof(defines_str), defines, num_defines, ':');
    join_strings(flags_str, sizeof(flags_str), flags, num_flags, ':');
    
    /* Symbol names */
    const char *sym_names[] = {
        "",  /* null symbol */
        "__armitage_sources",
        "__armitage_includes", 
        "__armitage_defines",
        "__armitage_flags",
        "__armitage_output",
    };
    const char *sym_values[] = {
        "",
        sources_str,
        includes_str,
        defines_str,
        flags_str,
        output_file ? output_file : "",
    };
    const int num_syms = 6;
    
    /* Build string table */
    char strtab[MAX_METADATA];
    size_t strtab_size = 1;  /* Start with null byte */
    strtab[0] = '\0';
    
    uint32_t sym_name_offsets[6];
    sym_name_offsets[0] = 0;  /* null symbol */
    
    for (int i = 1; i < num_syms; i++) {
        sym_name_offsets[i] = strtab_size;
        size_t len = strlen(sym_names[i]) + 1;
        memcpy(strtab + strtab_size, sym_names[i], len);
        strtab_size += len;
    }
    
    /* Build data section with symbol values */
    char data[MAX_METADATA];
    size_t data_size = 0;
    uint64_t sym_offsets[6];
    uint64_t sym_sizes[6];
    
    sym_offsets[0] = 0;
    sym_sizes[0] = 0;
    
    for (int i = 1; i < num_syms; i++) {
        sym_offsets[i] = data_size;
        size_t len = strlen(sym_values[i]) + 1;
        sym_sizes[i] = len;
        memcpy(data + data_size, sym_values[i], len);
        data_size += len;
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
    
    /* Sections: null, .shstrtab, .strtab, .symtab, .data */
    size_t data_off = ehdr_size;
    size_t shstrtab_off = data_off + data_size;
    size_t strtab_off = shstrtab_off + shstrtab_size;
    size_t symtab_off = strtab_off + strtab_size;
    size_t symtab_size = num_syms * sizeof(Elf64_Sym);
    size_t shdr_off = symtab_off + symtab_size;
    /* Align to 8 bytes */
    shdr_off = (shdr_off + 7) & ~7;
    
    /* Build ELF header */
    Elf64_Ehdr ehdr = {0};
    ehdr.e_ident[0] = 0x7f;
    ehdr.e_ident[1] = 'E';
    ehdr.e_ident[2] = 'L';
    ehdr.e_ident[3] = 'F';
    ehdr.e_ident[4] = 2;  /* 64-bit */
    ehdr.e_ident[5] = 1;  /* little endian */
    ehdr.e_ident[6] = 1;  /* ELF version */
    ehdr.e_type = ET_REL;
    ehdr.e_machine = EM_X86_64;
    ehdr.e_version = 1;
    ehdr.e_shoff = shdr_off;
    ehdr.e_ehsize = sizeof(Elf64_Ehdr);
    ehdr.e_shentsize = sizeof(Elf64_Shdr);
    ehdr.e_shnum = 5;
    ehdr.e_shstrndx = 1;  /* .shstrtab is section 1 */
    
    /* Build section headers */
    Elf64_Shdr shdrs[5] = {0};
    
    /* Section 0: null */
    
    /* Section 1: .shstrtab */
    shdrs[1].sh_name = sh_shstrtab;
    shdrs[1].sh_type = SHT_STRTAB;
    shdrs[1].sh_offset = shstrtab_off;
    shdrs[1].sh_size = shstrtab_size;
    shdrs[1].sh_addralign = 1;
    
    /* Section 2: .strtab */
    shdrs[2].sh_name = sh_strtab;
    shdrs[2].sh_type = SHT_STRTAB;
    shdrs[2].sh_offset = strtab_off;
    shdrs[2].sh_size = strtab_size;
    shdrs[2].sh_addralign = 1;
    
    /* Section 3: .symtab */
    shdrs[3].sh_name = sh_symtab;
    shdrs[3].sh_type = SHT_SYMTAB;
    shdrs[3].sh_offset = symtab_off;
    shdrs[3].sh_size = symtab_size;
    shdrs[3].sh_link = 2;  /* .strtab */
    shdrs[3].sh_info = 1;  /* first global symbol */
    shdrs[3].sh_addralign = 8;
    shdrs[3].sh_entsize = sizeof(Elf64_Sym);
    
    /* Section 4: .data */
    shdrs[4].sh_name = sh_data;
    shdrs[4].sh_type = 1;  /* SHT_PROGBITS */
    shdrs[4].sh_flags = 3;  /* SHF_WRITE | SHF_ALLOC */
    shdrs[4].sh_offset = data_off;
    shdrs[4].sh_size = data_size;
    shdrs[4].sh_addralign = 1;
    
    /* Build symbol table */
    Elf64_Sym syms[6] = {0};
    for (int i = 1; i < num_syms; i++) {
        syms[i].st_name = sym_name_offsets[i];
        syms[i].st_info = ELF64_ST_INFO(STB_GLOBAL, STT_OBJECT);
        syms[i].st_shndx = 4;  /* .data section */
        syms[i].st_value = sym_offsets[i];
        syms[i].st_size = sym_sizes[i];
    }
    
    /* Write the file */
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;
    
    write(fd, &ehdr, sizeof(ehdr));
    write(fd, data, data_size);
    write(fd, shstrtab, shstrtab_size);
    write(fd, strtab, strtab_size);
    write(fd, syms, symtab_size);
    
    /* Pad to section header alignment */
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

/* Determine output filename */
static const char *get_output_path(void) {
    if (output_file) return output_file;
    
    /* Default: first source with .o extension */
    if (num_sources > 0) {
        static char buf[MAX_PATH];
        const char *src = sources[0];
        const char *dot = strrchr(src, '.');
        const char *slash = strrchr(src, '/');
        const char *base = slash ? slash + 1 : src;
        size_t baselen = dot ? (size_t)(dot - base) : strlen(base);
        snprintf(buf, sizeof(buf), "%.*s.o", (int)baselen, base);
        return buf;
    }
    
    return "a.o";
}

int main(int argc, char **argv) {
    /* Log the invocation */
    log_invocation(argc, argv);
    
    /* Parse arguments */
    parse_args(argc, argv);
    
    /* Handle preprocessor-only mode */
    if (preprocess_only) {
        /* Just output nothing - cmake probes don't care */
        return 0;
    }
    
    /* Handle syntax-only mode */
    if (syntax_only) {
        return 0;
    }
    
    /* If no sources and no -c, might be a link - pass through for now */
    if (num_sources == 0 && !compile_only) {
        /* This is probably a link invocation - let ld-shim handle it */
        /* For now, just succeed */
        return 0;
    }
    
    /* Compile mode: emit fake .o */
    if (compile_only || num_sources > 0) {
        const char *out = get_output_path();
        if (write_fake_object(out) < 0) {
            fprintf(stderr, "cc-shim: failed to write %s\n", out);
            return 1;
        }
    }
    
    return 0;
}
