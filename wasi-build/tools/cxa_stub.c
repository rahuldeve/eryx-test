/* wasi-sdk's libc++abi is built without exception support, so nothing defines
 * these, yet numpy's C++ (np.unique, pocketfft) throws on errors it would
 * convert to Python exceptions. Without an unwinder no catch block can run
 * anyway: abort, as an uncaught exception would. Hidden, so each module that
 * pulls this in from the archive gets its own copy. */
#include <stdlib.h>

__attribute__((visibility("hidden"))) void *__cxa_allocate_exception(size_t size) {
    (void)size;
    abort();
}

__attribute__((visibility("hidden"))) void __cxa_throw(void *obj, void *tinfo, void (*dest)(void *)) {
    (void)obj;
    (void)tinfo;
    (void)dest;
    abort();
}
