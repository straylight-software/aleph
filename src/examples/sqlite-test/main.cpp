// SQLite3 test - show nix dep resolution works for database libs too

#include <stdio.h>
#include <sqlite3.h>

int main(void) {
    printf("=== SQLite3 test ===\n\n");
    printf("SQLite version: %s\n", sqlite3_libversion());
    printf("SQLite source ID: %s\n", sqlite3_sourceid());
    
    // Create in-memory database
    sqlite3 *db;
    int rc = sqlite3_open(":memory:", &db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        return 1;
    }
    printf("\nOpened in-memory database OK\n");
    
    // Create a table
    const char *sql = "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT);";
    char *err_msg = 0;
    rc = sqlite3_exec(db, sql, 0, 0, &err_msg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", err_msg);
        sqlite3_free(err_msg);
        sqlite3_close(db);
        return 1;
    }
    printf("Created table OK\n");
    
    // Insert some data
    sql = "INSERT INTO test (name) VALUES ('alice'), ('bob'), ('charlie');";
    rc = sqlite3_exec(db, sql, 0, 0, &err_msg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", err_msg);
        sqlite3_free(err_msg);
        sqlite3_close(db);
        return 1;
    }
    printf("Inserted 3 rows OK\n");
    
    // Query
    sqlite3_stmt *stmt;
    sql = "SELECT id, name FROM test;";
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare statement: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return 1;
    }
    
    printf("\nQuery results:\n");
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        int id = sqlite3_column_int(stmt, 0);
        const unsigned char *name = sqlite3_column_text(stmt, 1);
        printf("  id=%d, name=%s\n", id, name);
    }
    
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    
    printf("\n=== SQLite3 working ===\n");
    return 0;
}
