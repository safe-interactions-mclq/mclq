#include <mclq/alloc.h>
#include <stdlib.h>

static inline size_t align_up(size_t size, size_t alignment) {
	return (size + alignment - 1) & ~(alignment - 1);
}

void *stack_alloc(void *ctx, size_t size, size_t alignment) {
	StackAllocator *s = (StackAllocator *)ctx;
	uintptr_t current_addr = (uintptr_t)s->memory + s->offset;
	uintptr_t aligned_addr =
		(current_addr + (alignment - 1)) & ~(alignment - 1);
	size_t padding = aligned_addr - current_addr;

	if (s->offset + padding + size > s->size) {
		return NULL;
	}
	s->offset += padding + size;
	return (void *)aligned_addr;
}

void stack_reset(void *ctx) {
	((StackAllocator *)ctx)->offset = 0;
}

void *heap_alloc(void *ctx, size_t size, size_t alignment) {
	HeapAllocator *h = (HeapAllocator *)ctx;

	size = align_up(size, alignment);
	void *ptr = aligned_alloc(alignment, size);
	if (!ptr) {
		return NULL;
	}
	HeapNode *node = malloc(sizeof(HeapNode));
	if (!node) {
		free(ptr);
		return NULL;
	}
	node->ptr = ptr;
	node->next = h->head;
	h->head = node;
	return ptr;
}

void heap_reset(void *ctx) {
	HeapAllocator *h = (HeapAllocator *)ctx;
	HeapNode *current = h->head;

	while (current) {
		HeapNode *next = current->next;
		free(current->ptr);
		free(current);
		current = next;
	}
	h->head = NULL;
}

void *comp_alloc(void *ctx, size_t size, size_t alignment) {
	CompositeAllocator *c = (CompositeAllocator *)ctx;
	void *ptr = c->primary->alloc(c->primary->context, size, alignment);
	if (!ptr) {
		ptr = c->secondary->alloc(c->secondary->context, size, alignment);
	}
	return ptr;
}

void comp_reset(void *ctx) {
	CompositeAllocator *c = (CompositeAllocator *)ctx;
	c->primary->reset(c->primary->context);
	c->secondary->reset(c->secondary->context);
}

void *debug_alloc(void *ctx, size_t size, size_t alignment) {
	DebugAllocator *dbg = (DebugAllocator *)ctx;

	fprintf(dbg->log_stream,
			"[DebugAllocator: %s] Allocating %zu bytes (align: %zu)\n",
			dbg->name,
			size,
			alignment);

	void *ptr = dbg->delegate->alloc(dbg->delegate->context, size, alignment);

	if (ptr == NULL) {
		fprintf(dbg->log_stream,
				"[DebugAllocator: %s] FAILED to allocate!\n",
				dbg->name);
	} else {
		fprintf(dbg->log_stream,
				"[DebugAllocator: %s] Success: %p\n",
				dbg->name,
				ptr);
	}

	return ptr;
}

void debug_reset(void *ctx) {
	DebugAllocator *dbg = (DebugAllocator *)ctx;
	fprintf(dbg->log_stream, "[DebugAllocator: %s] Resetting...\n", dbg->name);
	dbg->delegate->reset(dbg->delegate->context);
}
