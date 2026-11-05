// Fake libnvidia-ml for developing nodestat without the real hardware.
//
//   cc -std=c23 -shared -fPIC -o mock_nvml.so mock_nvml.c -lm
//   NODESTAT_NVML_LIB=./mock_nvml.so MOCK_GPUS="NVIDIA GB300:2" ./nodestat
//
// MOCK_GPUS        "NAME:COUNT[,NAME:COUNT...]" (default "NVIDIA H100 80GB HBM3:8")
// MOCK_HANG_AFTER  seconds after init at which utilization queries hang forever,
//                  imitating a driver stuck on its rwsem.

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>

typedef struct { unsigned gpu, memory; } nv_util;
typedef struct { unsigned long long total, free, used; } nv_mem;
typedef struct {
    char busIdLegacy[16];
    unsigned domain, bus, device, pciDeviceId, pciSubSystemId;
    char busId[32];
} nv_pci;

struct dev {
    char name[96];
    int idx;
    double mem_gb, tdp, maxclk, memclk, slow;
    int ccmaj, ccmin, links, gen, width;
    bool ecc, fan;
};

static struct dev devs[64];
static int ndev;
static double t_init;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void profile(struct dev *d) {
    char n[96];
    strcpy(n, d->name);
    *d = (struct dev){.idx = d->idx, .mem_gb = 24, .tdp = 300, .maxclk = 2500, .memclk = 10000, .slow = 90,
                      .ccmaj = 8, .ccmin = 9, .gen = 4, .width = 16, .fan = true};
    strcpy(d->name, n);
    if (strstr(n, "GB300") || strstr(n, "B300")) {
        d->mem_gb = 288; d->tdp = 1400; d->maxclk = 2032; d->memclk = 3996; d->ccmaj = 10; d->ccmin = 3;
        d->links = 18; d->gen = 6; d->ecc = true; d->fan = false;
    } else if (strstr(n, "H100") || strstr(n, "H200")) {
        d->mem_gb = strstr(n, "H200") ? 141 : 80; d->tdp = 700; d->maxclk = 1980; d->memclk = 2619;
        d->ccmaj = 9; d->ccmin = 0; d->links = 18; d->gen = 5; d->ecc = true; d->fan = false;
    } else if (strstr(n, "A100")) {
        d->mem_gb = 80; d->tdp = 400; d->maxclk = 1410; d->memclk = 1593; d->ccmaj = 8; d->ccmin = 0;
        d->links = 12; d->ecc = true; d->fan = false;
    } else if (strstr(n, "PRO 6000")) {
        d->mem_gb = 96; d->tdp = 600; d->maxclk = 2617; d->memclk = 14001; d->ccmaj = 12; d->ccmin = 0;
        d->gen = 5; d->ecc = true;
    } else if (strstr(n, "4070")) {
        d->mem_gb = 12; d->tdp = 200; d->maxclk = 3105; d->memclk = 10501;
    }
}

int nvmlInit_v2(void) {
    const char *spec = getenv("MOCK_GPUS");
    char buf[1024];
    snprintf(buf, sizeof buf, "%s", spec && *spec ? spec : "NVIDIA H100 80GB HBM3:8");
    ndev = 0;
    for (char *save, *tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        char *colon = strrchr(tok, ':');
        int cnt = 1;
        if (colon) { *colon = 0; cnt = atoi(colon + 1); }
        for (int i = 0; i < cnt && ndev < 64; i++) {
            struct dev *d = &devs[ndev];
            d->idx = ndev++;
            snprintf(d->name, sizeof d->name, "%s", tok);
            profile(d);
        }
    }
    t_init = now();
    return 0;
}

const char *nvmlErrorString(int r) { return r ? "Not Supported" : "Success"; }
int nvmlSystemGetDriverVersion(char *s, unsigned n) { snprintf(s, n, "590.48.01"); return 0; }
int nvmlSystemGetCudaDriverVersion_v2(int *v) { *v = 13010; return 0; }
int nvmlDeviceGetCount_v2(unsigned *n) { *n = (unsigned)ndev; return 0; }
int nvmlDeviceGetHandleByIndex_v2(unsigned i, void **h) {
    if (i >= (unsigned)ndev) return 2;
    *h = &devs[i];
    return 0;
}

#define D ((struct dev *)h)

// A per-device load curve: ramps between phases so graphs have something to show.
static double load(void *h) {
    double t = now() - t_init;
    double phase = sin(t / 23.0 + D->idx * 0.7) * 0.5 + 0.5;
    return fmin(1, fmax(0, phase * 1.15 - 0.05 + 0.04 * sin(t * 1.7 + D->idx)));
}

int nvmlDeviceGetName(void *h, char *s, unsigned n) { snprintf(s, n, "%s", D->name); return 0; }
int nvmlDeviceGetPciInfo_v3(void *h, nv_pci *p) {
    memset(p, 0, sizeof *p);
    snprintf(p->busId, sizeof p->busId, "00000000:%02X:00.0", 0x18 + D->idx * 0x10);
    return 0;
}
int nvmlDeviceGetCudaComputeCapability(void *h, int *a, int *b) { *a = D->ccmaj; *b = D->ccmin; return 0; }
int nvmlDeviceGetVbiosVersion(void *h, char *s, unsigned n) { snprintf(s, n, "96.00.%02X.00.01", 0x60 + D->idx); return 0; }
int nvmlDeviceGetNumGpuCores(void *h, unsigned *v) { *v = D->ccmaj >= 9 ? 16896 : 5888; return 0; }
int nvmlDeviceGetMemoryBusWidth(void *h, unsigned *v) { *v = D->ecc ? 5120 : 192; return 0; }
int nvmlDeviceGetMaxClockInfo(void *h, int type, unsigned *v) {
    *v = (unsigned)(type == 2 ? D->memclk : D->maxclk);
    return 0;
}
int nvmlDeviceGetClockInfo(void *h, int type, unsigned *v) {
    double l = load(h);
    *v = (unsigned)(type == 2 ? (l > 0.05 ? D->memclk : 405) : 345 + (D->maxclk - 345) * fmin(1, l * 1.4) * (l > 0.9 ? 0.93 : 1));
    return 0;
}
int nvmlDeviceGetTemperatureThreshold(void *h, int t, unsigned *v) { *v = (unsigned)(t == 0 ? D->slow + 8 : D->slow); return 0; }
int nvmlDeviceGetTemperature(void *h, int t, unsigned *v) {
    (void)t;
    *v = (unsigned)(32 + load(h) * (D->slow - 40) + (D->idx == 3 ? 14 : 0));
    return 0;
}
int nvmlDeviceGetPowerManagementLimitConstraints(void *h, unsigned *a, unsigned *b) { *a = (unsigned)(D->tdp * 300); *b = (unsigned)(D->tdp * 1000); return 0; }
int nvmlDeviceGetEnforcedPowerLimit(void *h, unsigned *v) { *v = (unsigned)(D->tdp * 1000); return 0; }
int nvmlDeviceGetPowerUsage(void *h, unsigned *v) { *v = (unsigned)((0.1 + 0.88 * load(h)) * D->tdp * 1000); return 0; }
int nvmlDeviceGetTotalEnergyConsumption(void *h, unsigned long long *v) {
    *v = (unsigned long long)((now() - t_init + 86400) * D->tdp * 500);
    return 0;
}
int nvmlDeviceGetUtilizationRates(void *h, nv_util *u) {
    const char *hang = getenv("MOCK_HANG_AFTER");
    if (hang && now() - t_init > atof(hang)) for (;;) pause();
    double l = load(h);
    u->gpu = (unsigned)(l * 100);
    u->memory = (unsigned)(l * 62);
    return 0;
}
int nvmlDeviceGetMemoryInfo(void *h, nv_mem *m) {
    m->total = (unsigned long long)(D->mem_gb * (1ull << 30));
    double frac = load(h) > 0.1 ? 0.92 : 0.004;
    m->used = (unsigned long long)(m->total * frac);
    m->free = m->total - m->used;
    return 0;
}
int nvmlDeviceGetPerformanceState(void *h, int *p) { *p = load(h) > 0.05 ? 0 : 8; return 0; }
int nvmlDeviceGetFanSpeed(void *h, unsigned *v) {
    if (!D->fan) return 3;
    *v = (unsigned)(30 + load(h) * 50);
    return 0;
}
int nvmlDeviceGetCurrentClocksEventReasons(void *h, unsigned long long *r) {
    double l = load(h);
    *r = l < 0.05 ? 0x1 : l > 0.9 ? 0x4 : 0;
    if (D->idx == 3 && l > 0.8) *r |= 0x20;
    return 0;
}
int nvmlDeviceGetCurrPcieLinkGeneration(void *h, unsigned *v) { *v = load(h) > 0.05 ? (unsigned)D->gen : 1; return 0; }
int nvmlDeviceGetCurrPcieLinkWidth(void *h, unsigned *v) { *v = D->idx == 5 ? 8 : (unsigned)D->width; return 0; }
int nvmlDeviceGetMaxPcieLinkGeneration(void *h, unsigned *v) { *v = (unsigned)D->gen; return 0; }
int nvmlDeviceGetMaxPcieLinkWidth(void *h, unsigned *v) { *v = (unsigned)D->width; return 0; }
int nvmlDeviceGetPcieThroughput(void *h, int c, unsigned *v) {
    *v = (unsigned)(load(h) * (c ? 900000 : 120000));
    return 0;
}
int nvmlDeviceGetEncoderUtilization(void *h, unsigned *u, unsigned *p) { (void)h; *u = 0; *p = 167000; return 0; }
int nvmlDeviceGetDecoderUtilization(void *h, unsigned *u, unsigned *p) { (void)h; *u = 0; *p = 167000; return 0; }
int nvmlDeviceGetEccMode(void *h, int *c, int *p) { *c = *p = D->ecc; return 0; }
int nvmlDeviceGetTotalEccErrors(void *h, int type, int counter, unsigned long long *v) {
    *v = D->idx == 2 && type == 0 ? (counter ? 41 : 3) : 0;
    return 0;
}
int nvmlDeviceGetRemappedRows(void *h, unsigned *a, unsigned *b, unsigned *c, unsigned *d) {
    if (!D->ecc) return 3;
    *a = D->idx == 2 ? 2 : 0;
    *b = 0;
    *c = 0;
    *d = 0;
    return 0;
}
int nvmlDeviceGetNvLinkState(void *h, unsigned link, int *on) {
    if ((int)link >= D->links) return 2;
    *on = !(D->idx == 6 && link == 17);
    return 0;
}
int nvmlDeviceGetPersistenceMode(void *h, int *v) { *v = 1; (void)h; return 0; }
int nvmlDeviceGetMigMode(void *h, unsigned *a, unsigned *b) {
    if (!D->ecc) return 3;
    *a = *b = 0;
    return 0;
}
