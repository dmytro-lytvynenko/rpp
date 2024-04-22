#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
#include <math.h>

#define PRIME 101

typedef struct PatternNode {
    char* pattern;
    struct PatternNode* next;
} PatternNode;

PatternNode* createPatternNode(char* pattern) {
    PatternNode* newNode = malloc(sizeof(PatternNode));
    if (!newNode) {
        perror("Unable to allocate memory for a new pattern node");
        exit(1);
    }
    newNode->pattern = strdup(pattern);
    newNode->next = NULL;
    return newNode;
}

void appendPattern(PatternNode** head, char* pattern) {
    PatternNode* newNode = createPatternNode(pattern);
    if (*head == NULL) {
        *head = newNode;
    } else {
        PatternNode* current = *head;
        while (current->next != NULL) {
            current = current->next;
        }
        current->next = newNode;
    }
}

void freePatterns(PatternNode* head) {
    while (head != NULL) {
        PatternNode* tmp = head;
        head = head->next;
        free(tmp->pattern);
        free(tmp);
    }
}

long hash(char *str, int len) {
    long hash = 0;
    for (int i = 0; i < len; ++i) {
        hash = (hash * 256 + str[i]) % PRIME;
    }
    return hash;
}

int RabinKarpSearchSequential(char *pattern, char *text) {
    int patLen = strlen(pattern), textLen = strlen(text), count = 0;
    long patHash = hash(pattern, patLen), textHash = hash(text, patLen), h = 1;

    for (int k = 0; k < patLen - 1; k++) h = (h * 256) % PRIME;

    for (int i = 0; i <= textLen - patLen; ++i) {
        if (patHash == textHash && strncmp(pattern, &text[i], patLen) == 0) count++;
        if (i < textLen - patLen) {
            textHash = (textHash + PRIME - h * text[i] % PRIME) % PRIME;
            textHash = (textHash * 256 + text[i + patLen]) % PRIME;
        }
    }
    return count;
}

int ParallelRabinKarpSearch(char *pattern, char *text) {
    int patLen = strlen(pattern), textLen = strlen(text), count = 0;
    int numThreads;

    #pragma omp parallel reduction(+:count) private(numThreads)
    {
        numThreads = omp_get_num_threads();
        int threadID = omp_get_thread_num();
        int start = threadID * textLen / numThreads;
        int end = (threadID + 1) * textLen / numThreads;

        // Adjust end for the last thread to cover the entire text
        if (threadID == numThreads - 1) end = textLen;

        char* segment = malloc((end - start + patLen) * sizeof(char));
        strncpy(segment, text + start, end - start + patLen - 1);
        segment[end - start + patLen - 1] = '\0';

        count += RabinKarpSearchSequential(pattern, segment);

        free(segment);
    }

    return count;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <file_path>\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (!file) {
        perror("Error opening file");
        return 1;
    }

    PatternNode* patterns = NULL;
    char* text = NULL;
    size_t len = 0;
    ssize_t read;
    char* line = NULL;

    while ((read = getline(&line, &len, file)) != -1) {
        if (line[read - 1] == '\n') line[read - 1] = '\0';
        appendPattern(&patterns, line);
    }
    free(line);

    // Detach the last node to use its pattern as the text
    PatternNode* current = patterns, *prev = NULL;
    while (current && current->next) {
        prev = current;
        current = current->next;
    }
    if (current) {
        text = current->pattern;
        if (prev) prev->next = NULL;
        else patterns = NULL;
        free(current);
    }

    if (!text) {
        fprintf(stderr, "No text found in the file\n");
        fclose(file);
        return 1;
    }

    struct timeval start, end;

    gettimeofday(&start, NULL);
    current = patterns;
    while (current) {
        int count = RabinKarpSearchSequential(current->pattern, text);
        // printf("Pattern: %s\nOccurrences (Sequential): %d\n", current->pattern, count);
        current = current->next;
    }
    gettimeofday(&end, NULL);
    printf("Total execution time for sequential search: %f seconds\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0);

    gettimeofday(&start, NULL);
    current = patterns;
    while (current) {
        int count = ParallelRabinKarpSearch(current->pattern, text);
        // printf("Pattern: %s\nOccurrences (Parallel): %d\n", current->pattern, count);
        current = current->next;
    }
    gettimeofday(&end, NULL);
    printf("Total execution time for parallel search: %f seconds\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0);

    freePatterns(patterns);
    free(text);
    fclose(file);

    return 0;
}
