#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#define HASH_BASE 31
#define HASH_MOD 1000000007

int compute_hash(const char *str, int len) {
    int hash = 0;
    for (int i = 0; i < len; i++) {
        hash = (hash * HASH_BASE + str[i]) % HASH_MOD;
    }
    return hash;
}

char *read_line(FILE *file) {
    char *buffer = NULL;
    int capacity = 0;
    int n = 0;
    char c;

    while ((c = fgetc(file)) != '\n' && c != EOF) {
        if (n + 1 > capacity) {
            capacity = capacity ? capacity * 2 : 256;
            buffer = realloc(buffer, capacity);
        }
        buffer[n++] = c;
    }
    if (n == 0 && c == EOF) {
        free(buffer);
        return NULL;
    }
    buffer = realloc(buffer, n + 1);
    buffer[n] = '\0';
    return buffer;
}

void search_substrings(char *text, char *substring, int substring_len, int text_len, int *count) {
    int substring_hash = compute_hash(substring, substring_len);
    int text_hash = 0, base_pow = 1;
    
    for (int i = 0; i < substring_len - 1; i++) {
        base_pow = (base_pow * HASH_BASE) % HASH_MOD;
    }

    for (int i = 0; i <= text_len - substring_len; i++) {
        if (i == 0) {
            text_hash = compute_hash(text, substring_len);
        } else {
            text_hash = (text_hash - text[i-1] * base_pow % HASH_MOD + HASH_MOD) % HASH_MOD;
            text_hash = (text_hash * HASH_BASE + text[i + substring_len - 1]) % HASH_MOD;
        }

        if (text_hash == substring_hash && strncmp(text + i, substring, substring_len) == 0) {
            (*count)++;
        }
    }
}

int main(int argc, char *argv[]) {
    int pid, nodenum;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nodenum);

    double start_time, end_time;
    start_time = MPI_Wtime();

    int num_substrings = 0;
    char **substrings = NULL;
    char *text = NULL;
    int total_count = 0;

    if (pid == 0) {
        FILE *file = fopen(argv[1], "r");
        if (!file) {
            fprintf(stderr, "Failed to open file.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        substrings = malloc(sizeof(char*));
        while ((substrings[num_substrings] = read_line(file))) {
            num_substrings++;
            substrings = realloc(substrings, (num_substrings + 1) * sizeof(char*));
        }
        fclose(file);

        text = substrings[num_substrings - 1];
        substrings[num_substrings - 1] = NULL;
        num_substrings--;

        if (nodenum > 1) {
            int text_length = strlen(text) + 1;
            MPI_Bcast(&text_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(text, text_length, MPI_CHAR, 0, MPI_COMM_WORLD);
            MPI_Bcast(&num_substrings, 1, MPI_INT, 0, MPI_COMM_WORLD);

            for (int i = 0; i < num_substrings; i++) {
                int length = strlen(substrings[i]) + 1;
                MPI_Send(&length, 1, MPI_INT, i % nodenum, 0, MPI_COMM_WORLD);
                MPI_Send(substrings[i], length, MPI_CHAR, i % nodenum, 0, MPI_COMM_WORLD);
                free(substrings[i]);  // Free memory after sending
            }
        } else {
            for (int i = 0; i < num_substrings; i++) {
                int count = 0;
                search_substrings(text, substrings[i], strlen(substrings[i]), strlen(text), &count);
                total_count += count;
                free(substrings[i]);
            }
        }
        free(substrings);
    } else {
        int text_length;
        MPI_Bcast(&text_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
        text = malloc(text_length);
        MPI_Bcast(text, text_length, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num_substrings, 1, MPI_INT, 0, MPI_COMM_WORLD);

        for (int i = pid; i < num_substrings; i += nodenum) {
            int length;
            MPI_Recv(&length, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            char *substring = malloc(length);
            MPI_Recv(substring, length, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int count = 0;
            search_substrings(text, substring, length - 1, text_length - 1, &count);
            total_count += count;

            free(substring);
        }
    }

    if (nodenum > 1) {
        int reduced_count = 0;
        MPI_Reduce(&total_count, &reduced_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (pid == 0) {
            end_time = MPI_Wtime();
            printf("Execution time: %.6f seconds\nNumber of processes: %d\n", end_time - start_time, nodenum);
        }
    } else {
        end_time = MPI_Wtime();
        printf("Execution time: %.6f seconds\nNumber of processes: %d\n", end_time - start_time, nodenum);
    }

    if (pid != 0) {
        free(text);
    }

    MPI_Finalize();
    return 0;
}
