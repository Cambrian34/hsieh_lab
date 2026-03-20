#include <windows.h>

static LARGE_INTEGER _freq   = {0};
static LARGE_INTEGER _origin = {0};
static int           _ready  = 0;

__declspec(dllexport) int timer_init(void) {
    if (!QueryPerformanceFrequency(&_freq)) return 0;
    QueryPerformanceCounter(&_origin);
    _ready = 1;
    return 1;
}

__declspec(dllexport) long long timer_freq_hz(void) {
    return _ready ? _freq.QuadPart : 0LL;
}

__declspec(dllexport) long long timer_ticks(void) {
    LARGE_INTEGER now;
    if (!_ready) return -1LL;
    QueryPerformanceCounter(&now);
    return now.QuadPart - _origin.QuadPart;
}

__declspec(dllexport) double timer_seconds(void) {
    LARGE_INTEGER now;
    if (!_ready) return -1.0;
    QueryPerformanceCounter(&now);
    return (double)(now.QuadPart - _origin.QuadPart) / (double)_freq.QuadPart;
}

__declspec(dllexport) double timer_microseconds(void) {
    LARGE_INTEGER now;
    if (!_ready) return -1.0;
    QueryPerformanceCounter(&now);
    return (double)(now.QuadPart - _origin.QuadPart)
           * 1.0e6 / (double)_freq.QuadPart;
}

__declspec(dllexport) double timer_delta_us(long long t_start, long long t_end) {
    if (!_ready) return -1.0;
    return (double)(t_end - t_start) * 1.0e6 / (double)_freq.QuadPart;
}

#ifdef STANDALONE
#include <stdio.h>
int main(void) {
    long long t0, t1; int i;
    if (!timer_init()) { fprintf(stderr, "QPC unavailable\n"); return 1; }
    printf("Frequency: %lld Hz  |  Resolution: %.4f us\n\n",
           timer_freq_hz(), 1.0e6 / (double)timer_freq_hz());
    t0 = timer_ticks();
    for (i = 0; i < 10; i++) {
        double target = timer_microseconds() + 200.0;
        while (timer_microseconds() < target) {}   /* spin 200 us */
        t1 = timer_ticks();
        printf("sample %2d  elapsed: %10.2f us  delta: %8.2f us\n",
               i+1, timer_microseconds(), timer_delta_us(t0, t1));
        t0 = t1;
    }
    return 0;
}
#endif