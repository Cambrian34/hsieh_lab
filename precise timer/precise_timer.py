import ctypes, os, time

class PreciseTimer:
    def __init__(self, dll_path=None):
        if dll_path is None:
            dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "precise_timer.dll")
        self._lib = ctypes.CDLL(dll_path)
        self._lib.timer_init.restype          = ctypes.c_int
        self._lib.timer_freq_hz.restype       = ctypes.c_longlong
        self._lib.timer_ticks.restype         = ctypes.c_longlong
        self._lib.timer_seconds.restype       = ctypes.c_double
        self._lib.timer_microseconds.restype  = ctypes.c_double
        self._lib.timer_delta_us.restype      = ctypes.c_double
        self._lib.timer_delta_us.argtypes     = [ctypes.c_longlong, ctypes.c_longlong]
        if not self._lib.timer_init():
            raise RuntimeError("QueryPerformanceCounter unavailable.")

    def freq_hz(self):       return self._lib.timer_freq_hz()
    def resolution_us(self): return 1.0e6 / self.freq_hz()
    def ticks(self):         return self._lib.timer_ticks()
    def seconds(self):       return self._lib.timer_seconds()
    def microseconds(self):  return self._lib.timer_microseconds()
    def delta_us(self, t0, t1): return self._lib.timer_delta_us(t0, t1)

if __name__ == "__main__":
    t = PreciseTimer()
    print(f"Frequency : {t.freq_hz():,} Hz  |  Resolution: {t.resolution_us():.4f} us\n")
    t0 = t.ticks()
    for i in range(10):
        time.sleep(0.0002)          # ~200 us
        t1 = t.ticks()
        print(f"sample {i+1:2d}  elapsed: {t.microseconds():10.2f} us  "
              f"delta: {t.delta_us(t0,t1):8.2f} us")
        t0 = t1