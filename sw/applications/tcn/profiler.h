#pragma once
#ifndef TCN_PROFILER_H_
#define TCN_PROFILER_H_

#include <stdint.h>

#ifndef TCN_PROF_MAX_SECTIONS
#define TCN_PROF_MAX_SECTIONS 64
#endif

typedef struct {
    uint64_t total_ticks;
    uint32_t calls;
    uint64_t last_start;
    uint8_t active;
} ProfEntry;

typedef struct {
    ProfEntry sections[TCN_PROF_MAX_SECTIONS];
} Profiler;

#ifndef PROF_NOW
uint64_t prof_now(void);
#define PROF_NOW() prof_now()
#endif

static inline void profiler_reset(Profiler *p, unsigned section_count)
{
    if (!p) return;
    if (section_count > TCN_PROF_MAX_SECTIONS) {
        section_count = TCN_PROF_MAX_SECTIONS;
    }
    for (unsigned i = 0; i < section_count; ++i) {
        p->sections[i].total_ticks = 0;
        p->sections[i].calls = 0;
        p->sections[i].last_start = 0;
        p->sections[i].active = 0;
    }
}

static inline void profiler_begin(Profiler *p, unsigned id)
{
    if (!p || id >= TCN_PROF_MAX_SECTIONS) return;
    if (p->sections[id].active) return;
    p->sections[id].active = 1;
    p->sections[id].last_start = PROF_NOW();
}

static inline void profiler_end(Profiler *p, unsigned id)
{
    uint64_t now;
    ProfEntry *entry;

    if (!p || id >= TCN_PROF_MAX_SECTIONS) return;
    entry = &p->sections[id];
    if (!entry->active) return;

    now = PROF_NOW();
    entry->active = 0;
    entry->calls += 1;
    entry->total_ticks += (now - entry->last_start);
}

#endif // TCN_PROFILER_H_
