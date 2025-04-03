#ifndef MCLQ_ALLOC_H
#define MCLQ_ALLOC_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
	void *context;
	void *(*alloc)(void *context, size_t size, size_t alignment);
	void (*reset)(void *context);
} Allocator;

typedef struct {
	uint8_t *memory;
	size_t size;
	size_t offset;
} StackAllocator;

typedef struct HeapNode {
    void *ptr;
    struct HeapNode *next;
} HeapNode;

typedef struct {
    HeapNode *head;
} HeapAllocator;

typedef struct {
	Allocator *primary;
	Allocator *secondary;
} CompositeAllocator;

typedef struct {
    Allocator *delegate;
    FILE* log_stream;
    const char *name;
} DebugAllocator;

// NOTE: assumes that alignment is a power of 2
void *stack_alloc(void *ctx, size_t size, size_t alignment);
void *heap_alloc(void *ctx, size_t size, size_t alignment);
void *comp_alloc(void *ctx, size_t size, size_t alignment);

void stack_reset(void *ctx);
void heap_reset(void *ctx);
void comp_reset(void *ctx);

void *debug_alloc(void* ctx, size_t size, size_t alignment);
void debug_reset(void *ctx);

#endif
