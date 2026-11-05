// nodestat: a tiny status page for a GPU node.
//
// Serves one HTML page plus a JSON/SSE API with host stats (CPU, memory,
// cgroup limits, pressure stall info, disks, network) and NVIDIA GPU stats read
// straight from NVML via dlopen. It never runs nvidia-smi and never lists
// processes. NVML polling runs on its own thread, so a hung driver lock makes
// the GPU panel go stale instead of taking the page down.
//
// Build:  cc -std=c23 -O2 -pthread -o nodestat nodestat.c -ldl -lm
// Run:    nodestat [-p PORT] [-b ADDR] [--www DIR] | --once | --gpu-tag
//
// Environment:
//   NODESTAT_PORT      listen port (default 8080)
//   NODESTAT_TOKEN     if set, required as ?token= once (then a cookie)
//   NODE_NAME          display name; else the first line of /run/nodename;
//                      else gethostname()
//   NODESTAT_NVML_LIB  NVML library to load (default libnvidia-ml.so.1)

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

static const char INDEX_HTML[] = {
#embed "index.html"
    , 0};

// ---------------------------------------------------------------- utilities

static double now_mono(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static double now_wall(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void sleep_s(double s) {
    if (s <= 0) return;
    struct timespec ts = {(time_t)s, (long)((s - (time_t)s) * 1e9)};
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {}
}

// Reads a whole small file into buf. Returns length, or -1.
static ssize_t slurp(const char *path, char *buf, size_t cap) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    size_t n = 0;
    while (n + 1 < cap) {
        ssize_t k = read(fd, buf + n, cap - 1 - n);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) break;
        n += (size_t)k;
    }
    close(fd);
    buf[n] = 0;
    return (ssize_t)n;
}

// Reads one number from a file; "max" (cgroup unlimited) yields NAN.
static double read_num(const char *path) {
    char buf[64];
    if (slurp(path, buf, sizeof buf) <= 0 || !strncmp(buf, "max", 3)) return NAN;
    char *end;
    double v = strtod(buf, &end);
    return end == buf ? NAN : v;
}

static void copy_str(char *dst, size_t cap, const char *src) {
    size_t n = strnlen(src, cap - 1);
    memcpy(dst, src, n);
    dst[n] = 0;
}

// ---------------------------------------------------------- string builder

typedef struct {
    char *p;
    size_t n, cap;
} sb;

static void sb_reserve(sb *b, size_t extra) {
    if (b->n + extra + 1 <= b->cap) return;
    size_t c = b->cap ? b->cap : 512;
    while (c < b->n + extra + 1) c *= 2;
    char *np = realloc(b->p, c);
    if (!np) abort();
    b->p = np;
    b->cap = c;
}

static void sb_put(sb *b, const char *s, size_t len) {
    sb_reserve(b, len);
    memcpy(b->p + b->n, s, len);
    b->n += len;
    b->p[b->n] = 0;
}

static void sb_puts(sb *b, const char *s) { sb_put(b, s, strlen(s)); }

[[gnu::format(printf, 2, 3)]] static void sb_printf(sb *b, const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int k = vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    if (k > 0) {
        sb_reserve(b, (size_t)k);
        vsnprintf(b->p + b->n, (size_t)k + 1, fmt, ap2);
        b->n += (size_t)k;
    }
    va_end(ap2);
}

static void sb_free(sb *b) {
    free(b->p);
    *b = (sb){};
}

// JSON emission. Every value is followed by a comma; sb_end() strips the
// trailing comma before closing a container.
static void sb_end(sb *b, char close) {
    if (b->n && b->p[b->n - 1] == ',') b->n--;
    char s[2] = {close, ','};
    sb_put(b, s, 2);
}

static void sb_finish(sb *b) {
    if (b->n && b->p[b->n - 1] == ',') b->p[--b->n] = 0;
}

static void j_strval(sb *b, const char *s) {
    sb_puts(b, "\"");
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') sb_printf(b, "\\%c", c);
        else if (c < 0x20) sb_printf(b, "\\u%04x", c);
        else sb_put(b, (const char *)&c, 1);
    }
    sb_puts(b, "\",");
}

static void j_numval(sb *b, double v) {
    if (!isfinite(v)) sb_puts(b, "null,");
    else if (v == (double)(long long)v && fabs(v) < 9e15) sb_printf(b, "%lld,", (long long)v);
    else sb_printf(b, "%.15g,", v);
}

static void j_key(sb *b, const char *k) { sb_printf(b, "\"%s\":", k); }
static void j_num(sb *b, const char *k, double v) { j_key(b, k); j_numval(b, v); }
static void j_str(sb *b, const char *k, const char *s) { j_key(b, k); j_strval(b, s); }
static void j_obj(sb *b, const char *k) { if (k) j_key(b, k); sb_puts(b, "{"); }
static void j_arr(sb *b, const char *k) { if (k) j_key(b, k); sb_puts(b, "["); }

// ------------------------------------------------------------- host stats

#define MAXCPU 1024
#define MAXDISK 24

struct disk {
    char mount[128], fstype[24];
    double total, used, avail;
};

struct host {
    double uptime, load[3];
    double cpu_total; // % busy, averaged over the cores we may run on
    int ncores;
    float core[MAXCPU];  // % busy per allowed core
    double cg_cpu_used;  // cores used by our cgroup
    double cg_cpu_quota; // cores allowed by the cgroup quota, NAN if none
    double mem_total, mem_avail, swap_total, swap_free;
    double cg_mem_cur, cg_mem_max, cg_mem_anon; // NAN when unknown/unlimited
    double psi[3][2][2]; // [cpu,memory,io][some,full][avg10,avg60]
    int ndisk;
    struct disk disk[MAXDISK];
    double net_rx, net_tx, net_rx_total, net_tx_total;
};

struct host_static {
    char cpu_model[96], kernel[160], os[96];
    int ncpu_host;
};

static struct host_static g_hs;

struct cpu_prev {
    unsigned long long busy[MAXCPU], total[MAXCPU];
    double cg_usage_s, cg_t, net_t, net_rx, net_tx;
    bool have;
};

static void host_static_init(void) {
    struct utsname u;
    if (!uname(&u)) snprintf(g_hs.kernel, sizeof g_hs.kernel, "%s %s", u.sysname, u.release);
    static char buf[1 << 16];
    if (slurp("/proc/cpuinfo", buf, sizeof buf) > 0) {
        char *m = strstr(buf, "model name");
        if (m && (m = strchr(m, ':'))) {
            m += 2;
            char *e = strchr(m, '\n');
            if (e) *e = 0;
            copy_str(g_hs.cpu_model, sizeof g_hs.cpu_model, m);
        }
    }
    if (slurp("/etc/os-release", buf, sizeof buf) > 0) {
        char *m = strstr(buf, "PRETTY_NAME=");
        if (m) {
            m += 12;
            if (*m == '"') m++;
            char *e = strpbrk(m, "\"\n");
            if (e) *e = 0;
            copy_str(g_hs.os, sizeof g_hs.os, m);
        }
    }
    g_hs.ncpu_host = (int)sysconf(_SC_NPROCESSORS_CONF);
}

static void read_cpu(struct host *h, struct cpu_prev *pv) {
    static char buf[1 << 18];
    if (slurp("/proc/stat", buf, sizeof buf) <= 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    bool have_aff = !sched_getaffinity(0, sizeof set, &set);
    double sum = 0;
    int n = 0;
    for (char *line = buf; line && *line; line = strchr(line, '\n'), line = line ? line + 1 : nullptr) {
        if (strncmp(line, "cpu", 3) || !isdigit((unsigned char)line[3])) continue;
        int id;
        unsigned long long v[8] = {};
        if (sscanf(line, "cpu%d %llu %llu %llu %llu %llu %llu %llu %llu", &id, &v[0], &v[1], &v[2], &v[3],
                   &v[4], &v[5], &v[6], &v[7]) < 5 ||
            id < 0 || id >= MAXCPU)
            continue;
        unsigned long long total = 0;
        for (int i = 0; i < 8; i++) total += v[i];
        unsigned long long busy = total - v[3] - v[4];
        double pct = NAN;
        if (pv->have && total > pv->total[id])
            pct = 100.0 * (double)(busy - pv->busy[id]) / (double)(total - pv->total[id]);
        pv->busy[id] = busy;
        pv->total[id] = total;
        if (have_aff && !CPU_ISSET(id, &set)) continue;
        if (n < MAXCPU) h->core[n++] = (float)pct;
        if (isfinite(pct)) sum += pct;
    }
    h->ncores = n;
    h->cpu_total = pv->have && n ? sum / n : NAN;
}

// cgroup v2 first, then v1.
static void read_cgroup(struct host *h, struct cpu_prev *pv, double t) {
    char buf[4096];
    double usage_s = NAN;
    if (slurp("/sys/fs/cgroup/cpu.stat", buf, sizeof buf) > 0) {
        char *p = strstr(buf, "usage_usec ");
        if (p) usage_s = strtod(p + 11, nullptr) / 1e6;
    } else {
        double ns = read_num("/sys/fs/cgroup/cpuacct/cpuacct.usage");
        if (isfinite(ns)) usage_s = ns / 1e9;
    }
    h->cg_cpu_used = NAN;
    if (isfinite(usage_s) && isfinite(pv->cg_usage_s) && t > pv->cg_t)
        h->cg_cpu_used = (usage_s - pv->cg_usage_s) / (t - pv->cg_t);
    pv->cg_usage_s = usage_s;
    pv->cg_t = t;

    h->cg_cpu_quota = NAN;
    if (slurp("/sys/fs/cgroup/cpu.max", buf, sizeof buf) > 0) {
        double q, per;
        if (strncmp(buf, "max", 3) && sscanf(buf, "%lf %lf", &q, &per) == 2 && per > 0) h->cg_cpu_quota = q / per;
    } else {
        double q = read_num("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
        double per = read_num("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
        if (q > 0 && per > 0) h->cg_cpu_quota = q / per;
    }

    h->cg_mem_cur = read_num("/sys/fs/cgroup/memory.current");
    h->cg_mem_max = read_num("/sys/fs/cgroup/memory.max");
    h->cg_mem_anon = NAN;
    if (slurp("/sys/fs/cgroup/memory.stat", buf, sizeof buf) > 0) {
        char *p = strstr(buf, "anon ");
        if (p == buf || (p && p[-1] == '\n')) h->cg_mem_anon = strtod(p + 5, nullptr);
    }
    if (!isfinite(h->cg_mem_cur)) {
        h->cg_mem_cur = read_num("/sys/fs/cgroup/memory/memory.usage_in_bytes");
        h->cg_mem_max = read_num("/sys/fs/cgroup/memory/memory.limit_in_bytes");
    }
    // A limit at or above physical RAM is no limit.
    if (isfinite(h->cg_mem_max) && h->cg_mem_max >= h->mem_total) h->cg_mem_max = NAN;
}

static void read_meminfo(struct host *h) {
    char buf[8192];
    h->mem_total = h->mem_avail = h->swap_total = h->swap_free = NAN;
    if (slurp("/proc/meminfo", buf, sizeof buf) <= 0) return;
    static const struct {
        const char *key;
        size_t off;
    } keys[] = {
        {"MemTotal:", offsetof(struct host, mem_total)},
        {"MemAvailable:", offsetof(struct host, mem_avail)},
        {"SwapTotal:", offsetof(struct host, swap_total)},
        {"SwapFree:", offsetof(struct host, swap_free)},
    };
    for (size_t i = 0; i < sizeof keys / sizeof keys[0]; i++) {
        char *p = strstr(buf, keys[i].key);
        if (p) *(double *)((char *)h + keys[i].off) = strtod(p + strlen(keys[i].key), nullptr) * 1024;
    }
}

static void read_psi(struct host *h) {
    static const char *const res[3] = {"cpu", "memory", "io"};
    for (int r = 0; r < 3; r++) {
        for (int k = 0; k < 2; k++) h->psi[r][k][0] = h->psi[r][k][1] = NAN;
        char path[64], buf[512];
        snprintf(path, sizeof path, "/proc/pressure/%s", res[r]);
        if (slurp(path, buf, sizeof buf) <= 0) continue;
        for (char *line = buf; line && *line; line = strchr(line, '\n'), line = line ? line + 1 : nullptr) {
            double a10, a60;
            int k = !strncmp(line, "some", 4) ? 0 : !strncmp(line, "full", 4) ? 1 : -1;
            if (k >= 0 && sscanf(line + 4, " avg10=%lf avg60=%lf", &a10, &a60) == 2) {
                h->psi[r][k][0] = a10;
                h->psi[r][k][1] = a60;
            }
        }
    }
}

static bool skip_fstype(const char *t) {
    static const char *const skip[] = {
        "proc",     "sysfs",       "tmpfs",   "devtmpfs", "devpts",  "mqueue",    "cgroup",
        "cgroup2",  "securityfs",  "debugfs", "tracefs",  "pstore",  "bpf",       "configfs",
        "fusectl",  "hugetlbfs",   "autofs",  "nsfs",     "ramfs",   "efivarfs",  "selinuxfs",
        "binfmt_misc", "rpc_pipefs", "squashfs", "overlay_tmp", "nfsd", "fuse.portal", "fuse.gvfsd-fuse",
    };
    for (size_t i = 0; i < sizeof skip / sizeof skip[0]; i++)
        if (!strcmp(t, skip[i])) return true;
    return false;
}

static bool skip_mount(const char *m) {
    static const char *const pre[] = {"/proc", "/sys", "/dev", "/run", "/etc", "/usr", "/boot",
                                      "/var/lib/containers", "/var/lib/docker", "/snap", "/var/log"};
    if (!strcmp(m, "/")) return false;
    for (size_t i = 0; i < sizeof pre / sizeof pre[0]; i++) {
        size_t n = strlen(pre[i]);
        if (!strncmp(m, pre[i], n) && (m[n] == 0 || m[n] == '/')) return true;
    }
    return false;
}

// Undo the octal escapes /proc/self/mounts uses for spaces etc.
static void unescape_mount(char *s) {
    char *o = s;
    for (char *i = s; *i; i++) {
        if (i[0] == '\\' && i[1] >= '0' && i[1] <= '7' && i[2] && i[3]) {
            *o++ = (char)((i[1] - '0') * 64 + (i[2] - '0') * 8 + (i[3] - '0'));
            i += 3;
        } else {
            *o++ = *i;
        }
    }
    *o = 0;
}

static void read_disks(struct host *h) {
    static char buf[1 << 17];
    h->ndisk = 0;
    if (slurp("/proc/self/mounts", buf, sizeof buf) <= 0) return;
    dev_t seen[MAXDISK];
    for (char *save = nullptr, *line = strtok_r(buf, "\n", &save); line && h->ndisk < MAXDISK;
         line = strtok_r(nullptr, "\n", &save)) {
        char src[256], mnt[256], type[64];
        if (sscanf(line, "%255s %255s %63s", src, mnt, type) != 3) continue;
        unescape_mount(mnt);
        if (skip_fstype(type) || skip_mount(mnt)) continue;
        struct stat st;
        if (stat(mnt, &st) || !S_ISDIR(st.st_mode)) continue;
        struct statvfs vf;
        if (statvfs(mnt, &vf) || vf.f_blocks == 0) continue;
        // Bind mounts share st_dev; btrfs subvolumes don't, but report the
        // same block counts, so treat those as one filesystem too.
        bool dup = false;
        for (int i = 0; i < h->ndisk; i++)
            dup |= seen[i] == st.st_dev || (!strcmp(h->disk[i].fstype, type) &&
                                            h->disk[i].total == (double)vf.f_blocks * vf.f_frsize &&
                                            h->disk[i].used == (double)(vf.f_blocks - vf.f_bfree) * vf.f_frsize);
        if (dup) continue;
        struct disk *d = &h->disk[h->ndisk];
        seen[h->ndisk++] = st.st_dev;
        copy_str(d->mount, sizeof d->mount, mnt);
        copy_str(d->fstype, sizeof d->fstype, type);
        d->total = (double)vf.f_blocks * vf.f_frsize;
        d->avail = (double)vf.f_bavail * vf.f_frsize;
        d->used = (double)(vf.f_blocks - vf.f_bfree) * vf.f_frsize;
    }
}

static bool skip_iface(const char *n) {
    static const char *const pre[] = {"lo", "veth", "docker", "br-", "virbr", "podman", "cni", "flannel", "vnet"};
    for (size_t i = 0; i < sizeof pre / sizeof pre[0]; i++)
        if (!strncmp(n, pre[i], strlen(pre[i]))) return true;
    return false;
}

static void read_net(struct host *h, struct cpu_prev *pv, double t) {
    char buf[16384];
    h->net_rx = h->net_tx = h->net_rx_total = h->net_tx_total = NAN;
    if (slurp("/proc/net/dev", buf, sizeof buf) <= 0) return;
    double rx = 0, tx = 0;
    for (char *save = nullptr, *line = strtok_r(buf, "\n", &save); line; line = strtok_r(nullptr, "\n", &save)) {
        char *colon = strchr(line, ':');
        if (!colon) continue;
        *colon = 0;
        while (*line == ' ') line++;
        if (skip_iface(line)) continue;
        unsigned long long r, x, skip[7];
        if (sscanf(colon + 1, "%llu %llu %llu %llu %llu %llu %llu %llu %llu", &r, &skip[0], &skip[1], &skip[2],
                   &skip[3], &skip[4], &skip[5], &skip[6], &x) == 9) {
            rx += (double)r;
            tx += (double)x;
        }
    }
    h->net_rx_total = rx;
    h->net_tx_total = tx;
    if (pv->net_t > 0 && t > pv->net_t) {
        h->net_rx = fmax(0, (rx - pv->net_rx) / (t - pv->net_t));
        h->net_tx = fmax(0, (tx - pv->net_tx) / (t - pv->net_t));
    }
    pv->net_rx = rx;
    pv->net_tx = tx;
    pv->net_t = t;
}

static void host_sample(struct host *h, struct cpu_prev *pv) {
    double t = now_mono();
    char buf[256];
    h->uptime = NAN;
    if (slurp("/proc/uptime", buf, sizeof buf) > 0) h->uptime = strtod(buf, nullptr);
    h->load[0] = h->load[1] = h->load[2] = NAN;
    if (slurp("/proc/loadavg", buf, sizeof buf) > 0) sscanf(buf, "%lf %lf %lf", &h->load[0], &h->load[1], &h->load[2]);
    read_cpu(h, pv);
    read_meminfo(h);
    read_cgroup(h, pv, t);
    read_psi(h);
    read_disks(h);
    read_net(h, pv, t);
    pv->have = true;
}

// ------------------------------------------------------------------- NVML

typedef void *nvdev;
typedef struct { unsigned gpu, memory; } nv_util;
typedef struct { unsigned long long total, free, used; } nv_mem;
typedef struct {
    char busIdLegacy[16];
    unsigned domain, bus, device, pciDeviceId, pciSubSystemId;
    char busId[32];
} nv_pci;

enum { CLK_GRAPHICS = 0, CLK_SM = 1, CLK_MEM = 2 };
enum { TEMP_GPU = 0, THRESH_SHUTDOWN = 0, THRESH_SLOWDOWN = 1 };
enum { PCIE_TX = 0, PCIE_RX = 1 };
enum { ECC_CORRECTED = 0, ECC_UNCORRECTED = 1, ECC_VOLATILE = 0, ECC_AGGREGATE = 1 };

static struct {
    int (*Init)(void);
    const char *(*ErrorString)(int);
    int (*DriverVersion)(char *, unsigned);
    int (*CudaVersion)(int *);
    int (*Count)(unsigned *);
    int (*Handle)(unsigned, nvdev *);
    int (*Name)(nvdev, char *, unsigned);
    int (*Pci)(nvdev, nv_pci *);
    int (*CC)(nvdev, int *, int *);
    int (*Vbios)(nvdev, char *, unsigned);
    int (*Cores)(nvdev, unsigned *);
    int (*BusWidth)(nvdev, unsigned *);
    int (*MaxClock)(nvdev, int, unsigned *);
    int (*Clock)(nvdev, int, unsigned *);
    int (*TempThresh)(nvdev, int, unsigned *);
    int (*Temp)(nvdev, int, unsigned *);
    int (*PowerConstraints)(nvdev, unsigned *, unsigned *);
    int (*PowerLimit)(nvdev, unsigned *);
    int (*Power)(nvdev, unsigned *);
    int (*Energy)(nvdev, unsigned long long *);
    int (*Util)(nvdev, nv_util *);
    int (*Mem)(nvdev, nv_mem *);
    int (*Pstate)(nvdev, int *);
    int (*Fan)(nvdev, unsigned *);
    int (*Throttle)(nvdev, unsigned long long *);
    int (*PcieGen)(nvdev, unsigned *);
    int (*PcieWidth)(nvdev, unsigned *);
    int (*PcieGenMax)(nvdev, unsigned *);
    int (*PcieWidthMax)(nvdev, unsigned *);
    int (*PcieThroughput)(nvdev, int, unsigned *);
    int (*Enc)(nvdev, unsigned *, unsigned *);
    int (*Dec)(nvdev, unsigned *, unsigned *);
    int (*EccMode)(nvdev, int *, int *);
    int (*EccErrors)(nvdev, int, int, unsigned long long *);
    int (*Remapped)(nvdev, unsigned *, unsigned *, unsigned *, unsigned *);
    int (*NvLinkState)(nvdev, unsigned, int *);
    int (*Persistence)(nvdev, int *);
    int (*Mig)(nvdev, unsigned *, unsigned *);
} nv;

#define MAXGPU 16
#define MAXNVLINK 18

struct gpu {
    char name[96], short_name[32], bus[32], vbios[32], arch[16];
    int cc_major, cc_minor;
    double cores, bus_width, clk_sm_max, clk_mem_max, temp_slow, temp_shut, power_max;
    double pcie_gen_max, pcie_width_max, nvl_total;
    // dynamic
    double util, mem_util, mem_total, mem_used, temp, power, power_limit, energy;
    double clk_sm, clk_mem, clk_gr, pstate, fan;
    double pcie_gen, pcie_width, pcie_tx, pcie_rx, enc, dec;
    double ecc_on, ecc_corr, ecc_uncorr, ecc_corr_agg, ecc_uncorr_agg;
    double remap_corr, remap_unc, remap_pending, remap_fail;
    double nvl_active, persist, mig;
    bool have_throttle;
    unsigned long long throttle;
};

struct gpu_state {
    bool loaded;     // NVML initialised
    char err[160];   // why not, if not
    char driver[32], cuda[16];
    int n;
    struct gpu g[MAXGPU];
    double sweep_t;  // monotonic time of the last completed sweep
    double sweep_ms; // how long it took
    bool sweeping;
    double sweep_start;
};

static nvdev g_dev[MAXGPU];

static bool nvml_load(char *err, size_t errcap) {
    const char *lib = getenv("NODESTAT_NVML_LIB");
    void *h = dlopen(lib && *lib ? lib : "libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        snprintf(err, errcap, "no NVML: %s", dlerror());
        return false;
    }
#define SYM(field, a, b)                                              \
    do {                                                              \
        nv.field = (typeof(nv.field))dlsym(h, a);                     \
        if (!nv.field && (b)[0]) nv.field = (typeof(nv.field))dlsym(h, b); \
    } while (0)
    SYM(Init, "nvmlInit_v2", "nvmlInit");
    SYM(ErrorString, "nvmlErrorString", "");
    SYM(DriverVersion, "nvmlSystemGetDriverVersion", "");
    SYM(CudaVersion, "nvmlSystemGetCudaDriverVersion_v2", "nvmlSystemGetCudaDriverVersion");
    SYM(Count, "nvmlDeviceGetCount_v2", "nvmlDeviceGetCount");
    SYM(Handle, "nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex");
    SYM(Name, "nvmlDeviceGetName", "");
    SYM(Pci, "nvmlDeviceGetPciInfo_v3", "");
    SYM(CC, "nvmlDeviceGetCudaComputeCapability", "");
    SYM(Vbios, "nvmlDeviceGetVbiosVersion", "");
    SYM(Cores, "nvmlDeviceGetNumGpuCores", "");
    SYM(BusWidth, "nvmlDeviceGetMemoryBusWidth", "");
    SYM(MaxClock, "nvmlDeviceGetMaxClockInfo", "");
    SYM(Clock, "nvmlDeviceGetClockInfo", "");
    SYM(TempThresh, "nvmlDeviceGetTemperatureThreshold", "");
    SYM(Temp, "nvmlDeviceGetTemperature", "");
    SYM(PowerConstraints, "nvmlDeviceGetPowerManagementLimitConstraints", "");
    SYM(PowerLimit, "nvmlDeviceGetEnforcedPowerLimit", "");
    SYM(Power, "nvmlDeviceGetPowerUsage", "");
    SYM(Energy, "nvmlDeviceGetTotalEnergyConsumption", "");
    SYM(Util, "nvmlDeviceGetUtilizationRates", "");
    SYM(Mem, "nvmlDeviceGetMemoryInfo", "");
    SYM(Pstate, "nvmlDeviceGetPerformanceState", "");
    SYM(Fan, "nvmlDeviceGetFanSpeed", "");
    SYM(Throttle, "nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons");
    SYM(PcieGen, "nvmlDeviceGetCurrPcieLinkGeneration", "");
    SYM(PcieWidth, "nvmlDeviceGetCurrPcieLinkWidth", "");
    SYM(PcieGenMax, "nvmlDeviceGetMaxPcieLinkGeneration", "");
    SYM(PcieWidthMax, "nvmlDeviceGetMaxPcieLinkWidth", "");
    SYM(PcieThroughput, "nvmlDeviceGetPcieThroughput", "");
    SYM(Enc, "nvmlDeviceGetEncoderUtilization", "");
    SYM(Dec, "nvmlDeviceGetDecoderUtilization", "");
    SYM(EccMode, "nvmlDeviceGetEccMode", "");
    SYM(EccErrors, "nvmlDeviceGetTotalEccErrors", "");
    SYM(Remapped, "nvmlDeviceGetRemappedRows", "");
    SYM(NvLinkState, "nvmlDeviceGetNvLinkState", "");
    SYM(Persistence, "nvmlDeviceGetPersistenceMode", "");
    SYM(Mig, "nvmlDeviceGetMigMode", "");
#undef SYM
    if (!nv.Init || !nv.Count || !nv.Handle) {
        snprintf(err, errcap, "NVML is missing core symbols");
        return false;
    }
    int r = nv.Init();
    if (r) {
        snprintf(err, errcap, "nvmlInit failed: %s", nv.ErrorString ? nv.ErrorString(r) : "?");
        return false;
    }
    return true;
}

// "NVIDIA GeForce RTX 4070" -> "4070", "NVIDIA A100-SXM4-80GB" -> "A100",
// "NVIDIA RTX PRO 6000 Blackwell Server Edition" -> "PRO6000".
static void gpu_short_name(const char *full, char *out, size_t cap) {
    static const char *const drop[] = {"NVIDIA", "GeForce", "RTX", "GTX", "Tesla", "Quadro", "TITAN"};
    static const char *const stop[] = {"Generation", "Server", "Edition", "Workstation", "Max", "Blackwell",
                                       "Laptop", "GPU", "PCIe", "Founders", "Mobile"};
    char tmp[128];
    copy_str(tmp, sizeof tmp, full);
    size_t n = 0;
    out[0] = 0;
    for (char *save = nullptr, *tok = strtok_r(tmp, " -_", &save); tok; tok = strtok_r(nullptr, " -_", &save)) {
        bool skip = false, end = false;
        for (size_t i = 0; i < sizeof drop / sizeof drop[0]; i++) skip |= !strcasecmp(tok, drop[i]);
        // "TITAN" is the product itself when nothing follows
        if (skip && !strcasecmp(tok, "TITAN") && n == 0) skip = false;
        for (size_t i = 0; i < sizeof stop / sizeof stop[0]; i++) end |= !strcasecmp(tok, stop[i]);
        size_t L = strlen(tok);
        // memory sizes and form factors: 80GB, SXM4, HBM3
        if ((L > 2 && !strcasecmp(tok + L - 2, "GB") && isdigit((unsigned char)tok[0])) ||
            !strncasecmp(tok, "SXM", 3) || !strncasecmp(tok, "HBM", 3))
            end = true;
        if (end) break;
        if (skip) continue;
        if (n + L + 1 > cap) break;
        memcpy(out + n, tok, L + 1);
        n += L;
    }
    if (!n) copy_str(out, cap, "GPU");
}

static const char *arch_from_cc(int maj, int min) {
    switch (maj) {
    case 3: return "Kepler";
    case 5: return "Maxwell";
    case 6: return "Pascal";
    case 7: return min >= 5 ? "Turing" : "Volta";
    case 8: return min == 9 ? "Ada" : "Ampere";
    case 9: return "Hopper";
    case 10: case 11: case 12: return "Blackwell";
    default: return maj > 12 ? "post-Blackwell" : "?";
    }
}

#define Q(fn, ...) (nv.fn && nv.fn(__VA_ARGS__) == 0)

static void gpu_static(nvdev d, struct gpu *g) {
    unsigned a, b;
    int i1, i2;
    nv_pci pci;
    copy_str(g->name, sizeof g->name, "?");
    (void)Q(Name, d, g->name, sizeof g->name);
    gpu_short_name(g->name, g->short_name, sizeof g->short_name);
    if (Q(Pci, d, &pci)) copy_str(g->bus, sizeof g->bus, pci.busId);
    g->cc_major = g->cc_minor = -1;
    if (Q(CC, d, &i1, &i2)) {
        g->cc_major = i1;
        g->cc_minor = i2;
        copy_str(g->arch, sizeof g->arch, arch_from_cc(i1, i2));
    }
    (void)Q(Vbios, d, g->vbios, sizeof g->vbios);
    g->cores = Q(Cores, d, &a) ? a : NAN;
    g->bus_width = Q(BusWidth, d, &a) ? a : NAN;
    g->clk_sm_max = Q(MaxClock, d, CLK_SM, &a) ? a : NAN;
    g->clk_mem_max = Q(MaxClock, d, CLK_MEM, &a) ? a : NAN;
    g->temp_slow = Q(TempThresh, d, THRESH_SLOWDOWN, &a) ? a : NAN;
    g->temp_shut = Q(TempThresh, d, THRESH_SHUTDOWN, &a) ? a : NAN;
    g->power_max = Q(PowerConstraints, d, &a, &b) ? b / 1000.0 : NAN;
    g->pcie_gen_max = Q(PcieGenMax, d, &a) ? a : NAN;
    g->pcie_width_max = Q(PcieWidthMax, d, &a) ? a : NAN;
    int links = 0;
    for (unsigned l = 0; l < MAXNVLINK; l++) {
        int active;
        if (Q(NvLinkState, d, l, &active)) links++;
    }
    g->nvl_total = links ? links : NAN;
}

static void gpu_dynamic(nvdev d, struct gpu *g) {
    unsigned a, b, c, e;
    int i1, i2;
    unsigned long long ull;
    nv_util ut;
    nv_mem mem;
    g->util = g->mem_util = NAN;
    if (Q(Util, d, &ut)) {
        g->util = ut.gpu;
        g->mem_util = ut.memory;
    }
    g->mem_total = g->mem_used = NAN;
    if (Q(Mem, d, &mem)) {
        g->mem_total = (double)mem.total;
        g->mem_used = (double)mem.used;
    }
    g->temp = Q(Temp, d, TEMP_GPU, &a) ? a : NAN;
    g->power = Q(Power, d, &a) ? a / 1000.0 : NAN;
    g->power_limit = Q(PowerLimit, d, &a) ? a / 1000.0 : NAN;
    g->energy = Q(Energy, d, &ull) ? ull / 1000.0 : NAN; // joules
    g->clk_sm = Q(Clock, d, CLK_SM, &a) ? a : NAN;
    g->clk_mem = Q(Clock, d, CLK_MEM, &a) ? a : NAN;
    g->clk_gr = Q(Clock, d, CLK_GRAPHICS, &a) ? a : NAN;
    g->pstate = Q(Pstate, d, &i1) && i1 != 32 ? i1 : NAN;
    g->fan = Q(Fan, d, &a) ? a : NAN;
    g->have_throttle = Q(Throttle, d, &g->throttle);
    g->pcie_gen = Q(PcieGen, d, &a) ? a : NAN;
    g->pcie_width = Q(PcieWidth, d, &a) ? a : NAN;
    g->pcie_tx = Q(PcieThroughput, d, PCIE_TX, &a) ? a * 1024.0 : NAN; // bytes/s
    g->pcie_rx = Q(PcieThroughput, d, PCIE_RX, &a) ? a * 1024.0 : NAN;
    g->enc = Q(Enc, d, &a, &b) ? a : NAN;
    g->dec = Q(Dec, d, &a, &b) ? a : NAN;
    g->ecc_on = Q(EccMode, d, &i1, &i2) ? i1 : NAN;
    g->ecc_corr = g->ecc_uncorr = g->ecc_corr_agg = g->ecc_uncorr_agg = NAN;
    if (g->ecc_on == 1) {
        if (Q(EccErrors, d, ECC_CORRECTED, ECC_VOLATILE, &ull)) g->ecc_corr = (double)ull;
        if (Q(EccErrors, d, ECC_UNCORRECTED, ECC_VOLATILE, &ull)) g->ecc_uncorr = (double)ull;
        if (Q(EccErrors, d, ECC_CORRECTED, ECC_AGGREGATE, &ull)) g->ecc_corr_agg = (double)ull;
        if (Q(EccErrors, d, ECC_UNCORRECTED, ECC_AGGREGATE, &ull)) g->ecc_uncorr_agg = (double)ull;
    }
    g->remap_corr = g->remap_unc = g->remap_pending = g->remap_fail = NAN;
    if (Q(Remapped, d, &a, &b, &c, &e)) {
        g->remap_corr = a;
        g->remap_unc = b;
        g->remap_pending = c;
        g->remap_fail = e;
    }
    g->nvl_active = NAN;
    if (isfinite(g->nvl_total)) {
        int act = 0;
        for (unsigned l = 0; l < MAXNVLINK; l++) {
            int on;
            if (Q(NvLinkState, d, l, &on) && on) act++;
        }
        g->nvl_active = act;
    }
    g->persist = Q(Persistence, d, &i1) ? i1 : NAN;
    g->mig = Q(Mig, d, &a, &b) ? a : NAN;
}

static struct gpu_state g_gpu;
static pthread_mutex_t g_gpu_mu = PTHREAD_MUTEX_INITIALIZER;

static void *gpu_thread(void *arg) {
    (void)arg;
    static struct gpu_state local; // static: too large for a comfortable stack
    for (;;) {
        double t0 = now_mono();
        pthread_mutex_lock(&g_gpu_mu);
        g_gpu.sweeping = true;
        g_gpu.sweep_start = t0;
        pthread_mutex_unlock(&g_gpu_mu);
        if (!local.loaded) {
            local = (struct gpu_state){};
            if (nvml_load(local.err, sizeof local.err)) {
                local.loaded = true;
                (void)Q(DriverVersion, local.driver, sizeof local.driver);
                int cv;
                if (Q(CudaVersion, &cv)) snprintf(local.cuda, sizeof local.cuda, "%d.%d", cv / 1000, cv % 1000 / 10);
                unsigned cnt = 0;
                (void)Q(Count, &cnt);
                local.n = cnt > MAXGPU ? MAXGPU : (int)cnt;
                for (int i = 0; i < local.n; i++) {
                    if (!Q(Handle, (unsigned)i, &g_dev[i])) g_dev[i] = nullptr;
                    if (g_dev[i]) gpu_static(g_dev[i], &local.g[i]);
                }
            }
        }
        if (local.loaded)
            for (int i = 0; i < local.n; i++)
                if (g_dev[i]) gpu_dynamic(g_dev[i], &local.g[i]);
        double t1 = now_mono();
        pthread_mutex_lock(&g_gpu_mu);
        local.sweep_t = t1;
        local.sweep_ms = (t1 - t0) * 1000;
        local.sweeping = false;
        g_gpu = local;
        pthread_mutex_unlock(&g_gpu_mu);
        // Retry a missing driver slowly; otherwise sample once a second.
        sleep_s((local.loaded ? 1.0 : 15.0) - (now_mono() - t0));
    }
    return nullptr;
}

static struct host g_host;
static pthread_mutex_t g_host_mu = PTHREAD_MUTEX_INITIALIZER;

static void *host_thread(void *arg) {
    (void)arg;
    static struct cpu_prev pv = {.cg_usage_s = NAN};
    static struct host local;
    for (;;) {
        double t0 = now_mono();
        host_sample(&local, &pv);
        pthread_mutex_lock(&g_host_mu);
        g_host = local;
        pthread_mutex_unlock(&g_host_mu);
        sleep_s(1.0 - (now_mono() - t0));
    }
    return nullptr;
}

// ----------------------------------------------------------- snapshot JSON

static char g_name[96];

static void resolve_name(void) {
    const char *env = getenv("NODE_NAME");
    if (env && *env) {
        copy_str(g_name, sizeof g_name, env);
        return;
    }
    char buf[128];
    if (slurp("/run/nodename", buf, sizeof buf) > 0) {
        buf[strcspn(buf, "\r\n")] = 0;
        if (*buf) {
            copy_str(g_name, sizeof g_name, buf);
            return;
        }
    }
    if (gethostname(buf, sizeof buf)) copy_str(buf, sizeof buf, "node");
    buf[sizeof buf - 1] = 0;
    copy_str(g_name, sizeof g_name, buf);
}

// Counts of each GPU model in index order: "4x4070", "2xH100+1x4090".
static void gpu_tag(const struct gpu_state *gs, char *out, size_t cap) {
    out[0] = 0;
    if (!gs->loaded || gs->n == 0) {
        copy_str(out, cap, "cpu");
        return;
    }
    bool done[MAXGPU] = {};
    size_t n = 0;
    for (int i = 0; i < gs->n; i++) {
        if (done[i]) continue;
        int cnt = 0;
        for (int j = i; j < gs->n; j++)
            if (!done[j] && !strcmp(gs->g[i].short_name, gs->g[j].short_name)) done[j] = true, cnt++;
        int k = snprintf(out + n, cap - n, "%s%dx%s", n ? "+" : "", cnt, gs->g[i].short_name);
        if (k < 0 || (size_t)k >= cap - n) break;
        n += (size_t)k;
    }
}

static const struct {
    unsigned long long bit;
    const char *key;
} THROTTLE[] = {
    {0x1, "idle"},          {0x2, "app_clocks"},  {0x4, "power_cap"},  {0x8, "hw_slowdown"},
    {0x10, "sync_boost"},   {0x20, "sw_thermal"}, {0x40, "hw_thermal"}, {0x80, "power_brake"},
    {0x100, "display_clocks"},
};

// Effective memory use: our cgroup's if limited, else the host's.
static double mem_used_eff(const struct host *h) {
    return isfinite(h->cg_mem_max) && isfinite(h->cg_mem_cur) ? h->cg_mem_cur : h->mem_total - h->mem_avail;
}

#define HIST_GPU_FIELDS 5

// One compact history row: [t, cpu%, mem, psi_mem_some10, rx, tx, [[util, mem, W, °C, MHz], ...]].
static void emit_row(sb *b, double t, const struct host *h, const struct gpu_state *gs, bool gpu_fresh) {
    j_arr(b, nullptr);
    j_numval(b, round(t * 10) / 10);
    j_numval(b, round(h->cpu_total * 10) / 10);
    j_numval(b, mem_used_eff(h));
    j_numval(b, h->psi[1][0][0]);
    j_numval(b, round(h->net_rx));
    j_numval(b, round(h->net_tx));
    j_arr(b, nullptr);
    for (int i = 0; i < gs->n; i++) {
        const struct gpu *g = &gs->g[i];
        j_arr(b, nullptr);
        double v[HIST_GPU_FIELDS] = {g->util, g->mem_used, round(g->power * 10) / 10, g->temp, g->clk_sm};
        for (int k = 0; k < HIST_GPU_FIELDS; k++) j_numval(b, gpu_fresh ? v[k] : NAN);
        sb_end(b, ']');
    }
    sb_end(b, ']');
    sb_end(b, ']');
}

static void emit_snapshot(sb *b, sb *row) {
    static struct host h;
    static struct gpu_state gs;
    pthread_mutex_lock(&g_host_mu);
    h = g_host;
    pthread_mutex_unlock(&g_host_mu);
    pthread_mutex_lock(&g_gpu_mu);
    gs = g_gpu;
    pthread_mutex_unlock(&g_gpu_mu);
    double t = now_wall(), m = now_mono();
    // Stale: no sweep has finished for a while, or one is stuck in progress.
    double age = gs.sweep_t > 0 ? m - gs.sweep_t : NAN;
    if (gs.sweeping && m - gs.sweep_start > 3) age = fmax(isfinite(age) ? age : 0, m - gs.sweep_start);
    bool fresh = gs.loaded && isfinite(age) && age < 5;

    resolve_name();
    char tag[160];
    gpu_tag(&gs, tag, sizeof tag);

    j_obj(b, nullptr);
    j_num(b, "t", round(t * 1000) / 1000);
    j_obj(b, "node");
    j_str(b, "name", g_name);
    j_str(b, "gpu_tag", tag);
    j_str(b, "os", g_hs.os);
    j_str(b, "kernel", g_hs.kernel);
    j_str(b, "cpu_model", g_hs.cpu_model);
    j_num(b, "ncpu_host", g_hs.ncpu_host);
    j_str(b, "driver", gs.driver);
    j_str(b, "cuda", gs.cuda);
    sb_end(b, '}');

    j_obj(b, "host");
    j_num(b, "uptime", h.uptime);
    j_arr(b, "load");
    for (int i = 0; i < 3; i++) j_numval(b, h.load[i]);
    sb_end(b, ']');
    j_num(b, "cpu", round(h.cpu_total * 10) / 10);
    j_arr(b, "cores");
    for (int i = 0; i < h.ncores; i++) j_numval(b, round(h.core[i]));
    sb_end(b, ']');
    j_num(b, "cg_cpu_used", round(h.cg_cpu_used * 100) / 100);
    j_num(b, "cg_cpu_quota", h.cg_cpu_quota);
    j_num(b, "mem_total", h.mem_total);
    j_num(b, "mem_avail", h.mem_avail);
    j_num(b, "swap_total", h.swap_total);
    j_num(b, "swap_free", h.swap_free);
    j_num(b, "cg_mem_cur", h.cg_mem_cur);
    j_num(b, "cg_mem_max", h.cg_mem_max);
    j_num(b, "cg_mem_anon", h.cg_mem_anon);
    j_num(b, "mem_used", mem_used_eff(&h));
    j_num(b, "mem_limit", isfinite(h.cg_mem_max) ? h.cg_mem_max : h.mem_total);
    j_obj(b, "psi");
    static const char *const res[3] = {"cpu", "memory", "io"};
    for (int r = 0; r < 3; r++) {
        j_obj(b, res[r]);
        j_arr(b, "some");
        j_numval(b, h.psi[r][0][0]);
        j_numval(b, h.psi[r][0][1]);
        sb_end(b, ']');
        j_arr(b, "full");
        j_numval(b, h.psi[r][1][0]);
        j_numval(b, h.psi[r][1][1]);
        sb_end(b, ']');
        sb_end(b, '}');
    }
    sb_end(b, '}');
    j_arr(b, "disks");
    for (int i = 0; i < h.ndisk; i++) {
        j_obj(b, nullptr);
        j_str(b, "mount", h.disk[i].mount);
        j_str(b, "fs", h.disk[i].fstype);
        j_num(b, "total", h.disk[i].total);
        j_num(b, "used", h.disk[i].used);
        j_num(b, "avail", h.disk[i].avail);
        sb_end(b, '}');
    }
    sb_end(b, ']');
    j_obj(b, "net");
    j_num(b, "rx", round(h.net_rx));
    j_num(b, "tx", round(h.net_tx));
    j_num(b, "rx_total", h.net_rx_total);
    j_num(b, "tx_total", h.net_tx_total);
    sb_end(b, '}');
    sb_end(b, '}');

    j_obj(b, "gpu");
    j_key(b, "loaded");
    sb_puts(b, gs.loaded ? "true," : "false,");
    j_str(b, "error", gs.err);
    j_num(b, "age", isfinite(age) ? round(age * 10) / 10 : NAN);
    j_key(b, "fresh");
    sb_puts(b, fresh ? "true," : "false,");
    j_num(b, "sweep_ms", round(gs.sweep_ms));
    j_arr(b, "devices");
    for (int i = 0; i < gs.n; i++) {
        const struct gpu *g = &gs.g[i];
        j_obj(b, nullptr);
        j_num(b, "index", i);
        j_str(b, "name", g->name);
        j_str(b, "short", g->short_name);
        j_str(b, "bus", g->bus);
        j_str(b, "arch", g->arch);
        j_str(b, "vbios", g->vbios);
        if (g->cc_major >= 0) {
            char cc[16];
            snprintf(cc, sizeof cc, "%d.%d", g->cc_major, g->cc_minor);
            j_str(b, "cc", cc);
        }
        const struct { const char *k; double v; } nums[] = {
            {"cores", g->cores}, {"bus_width", g->bus_width}, {"clk_sm_max", g->clk_sm_max},
            {"clk_mem_max", g->clk_mem_max}, {"temp_slow", g->temp_slow}, {"temp_shut", g->temp_shut},
            {"power_max", g->power_max}, {"pcie_gen_max", g->pcie_gen_max}, {"pcie_width_max", g->pcie_width_max},
            {"nvl_total", g->nvl_total}, {"util", g->util}, {"mem_util", g->mem_util},
            {"mem_total", g->mem_total}, {"mem_used", g->mem_used}, {"temp", g->temp},
            {"power", round(g->power * 10) / 10}, {"power_limit", g->power_limit}, {"energy", round(g->energy)},
            {"clk_sm", g->clk_sm}, {"clk_mem", g->clk_mem}, {"clk_gr", g->clk_gr}, {"pstate", g->pstate},
            {"fan", g->fan}, {"pcie_gen", g->pcie_gen}, {"pcie_width", g->pcie_width}, {"pcie_tx", g->pcie_tx},
            {"pcie_rx", g->pcie_rx}, {"enc", g->enc}, {"dec", g->dec}, {"ecc_on", g->ecc_on},
            {"ecc_corr", g->ecc_corr}, {"ecc_uncorr", g->ecc_uncorr}, {"ecc_corr_agg", g->ecc_corr_agg},
            {"ecc_uncorr_agg", g->ecc_uncorr_agg}, {"remap_corr", g->remap_corr}, {"remap_unc", g->remap_unc},
            {"remap_pending", g->remap_pending}, {"remap_fail", g->remap_fail}, {"nvl_active", g->nvl_active},
            {"persist", g->persist}, {"mig", g->mig},
        };
        for (size_t k = 0; k < sizeof nums / sizeof nums[0]; k++) j_num(b, nums[k].k, nums[k].v);
        j_key(b, "throttle");
        if (g->have_throttle) {
            j_arr(b, nullptr);
            for (size_t k = 0; k < sizeof THROTTLE / sizeof THROTTLE[0]; k++)
                if (g->throttle & THROTTLE[k].bit) j_strval(b, THROTTLE[k].key);
            sb_end(b, ']');
        } else {
            sb_puts(b, "null,");
        }
        sb_end(b, '}');
    }
    sb_end(b, ']');
    sb_end(b, '}');

    if (row) {
        emit_row(row, t, &h, &gs, fresh);
        sb_finish(row);
        j_key(b, "row");
        sb_put(b, row->p, row->n);
        sb_puts(b, ",");
    }
    sb_end(b, '}');
    sb_finish(b);
}

// ------------------------------------------------------------ HTTP server

#define HIST_LEN 900 // 15 minutes at 1 Hz
#define MAXCLIENT 64
#define MAX_SSE_BACKLOG (4 << 20)

static char *g_hist[HIST_LEN];
static int g_hist_head, g_hist_n;
static sb g_latest;
static char g_token[128];
static const char *g_www;

struct client {
    int fd;
    bool sse, close_after;
    double t0;
    char in[8192];
    size_t inlen;
    sb out;
    size_t off;
};

static struct client g_cl[MAXCLIENT];
static int g_ncl;

static void cl_close(int i) {
    close(g_cl[i].fd);
    sb_free(&g_cl[i].out);
    g_cl[i] = g_cl[--g_ncl];
    g_cl[g_ncl] = (struct client){};
}

static void respond(struct client *c, int code, const char *status, const char *ctype, const char *extra,
                    const char *body, size_t len) {
    sb_printf(&c->out,
              "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nCache-Control: no-store\r\n"
              "X-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\nConnection: close\r\n%s\r\n",
              code, status, ctype, len, extra ? extra : "");
    sb_put(&c->out, body, len);
    c->close_after = true;
}

static bool token_eq(const char *a, size_t alen) {
    size_t blen = strlen(g_token);
    unsigned char diff = alen != blen;
    for (size_t i = 0; i < alen && i < blen; i++) diff |= (unsigned char)(a[i] ^ g_token[i]);
    return !diff;
}

// Finds `name=` in a query string or cookie header; returns value length.
static size_t find_param(const char *s, const char *end, const char *name, char sep, const char **val) {
    size_t nl = strlen(name);
    for (const char *p = s; p && p < end;) {
        while (p < end && (*p == ' ' || *p == sep)) p++;
        const char *e = memchr(p, sep, (size_t)(end - p));
        if (!e) e = end;
        if ((size_t)(e - p) > nl && !strncmp(p, name, nl) && p[nl] == '=') {
            *val = p + nl + 1;
            return (size_t)(e - *val);
        }
        p = e < end ? e + 1 : nullptr;
    }
    return 0;
}

static void serve_history(struct client *c) {
    sb b = {};
    sb_puts(&b, "[");
    for (int i = 0; i < g_hist_n; i++) {
        int k = (g_hist_head - g_hist_n + i + HIST_LEN) % HIST_LEN;
        sb_puts(&b, g_hist[k]);
        sb_puts(&b, ",");
    }
    sb_end(&b, ']');
    sb_finish(&b);
    respond(c, 200, "OK", "application/json", nullptr, b.p, b.n);
    sb_free(&b);
}

static void serve_index(struct client *c) {
    static const char csp[] = "Content-Security-Policy: default-src 'self'; style-src 'self' 'unsafe-inline'; "
                              "script-src 'self' 'unsafe-inline'; img-src 'self' data:\r\n";
    if (g_www) {
        char path[4096], *buf = malloc(1 << 20);
        snprintf(path, sizeof path, "%s/index.html", g_www);
        ssize_t n = buf ? slurp(path, buf, 1 << 20) : -1;
        if (n >= 0) respond(c, 200, "OK", "text/html; charset=utf-8", csp, buf, (size_t)n);
        else respond(c, 500, "Error", "text/plain", nullptr, "cannot read --www index.html\n", 29);
        free(buf);
        return;
    }
    respond(c, 200, "OK", "text/html; charset=utf-8", csp, INDEX_HTML, sizeof INDEX_HTML - 1);
}

static void handle_request(struct client *c) {
    char *hdr_end = strstr(c->in, "\r\n\r\n");
    char *sp1 = strchr(c->in, ' ');
    char *sp2 = sp1 ? strchr(sp1 + 1, ' ') : nullptr;
    if (!sp1 || !sp2 || sp2 > hdr_end) {
        respond(c, 400, "Bad Request", "text/plain", nullptr, "bad request\n", 12);
        return;
    }
    bool head = !strncmp(c->in, "HEAD ", 5);
    if (strncmp(c->in, "GET ", 4) && !head) {
        respond(c, 405, "Method Not Allowed", "text/plain", "Allow: GET\r\n", "GET only\n", 9);
        return;
    }
    char *path = sp1 + 1, *pend = sp2;
    char *q = memchr(path, '?', (size_t)(pend - path));
    char *path_end = q ? q : pend;
    size_t plen = (size_t)(path_end - path);
#define ROUTE(s) (plen == sizeof(s) - 1 && !strncmp(path, s, plen))

    if (ROUTE("/healthz")) {
        respond(c, 200, "OK", "text/plain", nullptr, "ok\n", 3);
        goto done;
    }
    if (g_token[0]) {
        const char *v;
        size_t vl = q ? find_param(q + 1, pend, "token", '&', &v) : 0;
        if (vl && token_eq(v, vl)) {
            // Swap the URL token for a cookie and drop it from the address bar.
            char extra[512];
            snprintf(extra, sizeof extra,
                     "Set-Cookie: nodestat_token=%s; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000\r\n"
                     "Location: %.*s\r\n",
                     g_token, (int)plen, path);
            respond(c, 303, "See Other", "text/plain", extra, "", 0);
            goto done;
        }
        bool ok = false;
        for (char *h = strcasestr(c->in, "\r\ncookie:"); h && h < hdr_end; h = strcasestr(h + 2, "\r\ncookie:")) {
            char *ve = strstr(h + 2, "\r\n");
            vl = find_param(h + 9, ve, "nodestat_token", ';', &v);
            if (vl && token_eq(v, vl)) ok = true;
        }
        if (!ok) {
            respond(c, 401, "Unauthorized", "text/plain", nullptr, "open this page with ?token=...\n", 31);
            goto done;
        }
    }
    if (ROUTE("/")) serve_index(c);
    else if (ROUTE("/api/snapshot")) respond(c, 200, "OK", "application/json", nullptr, g_latest.p, g_latest.n);
    else if (ROUTE("/api/history")) serve_history(c);
    else if (ROUTE("/api/events")) {
        sb_puts(&c->out, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\n"
                         "X-Accel-Buffering: no\r\nConnection: keep-alive\r\n\r\nretry: 2000\n\n");
        if (g_latest.n) {
            sb_puts(&c->out, "data: ");
            sb_put(&c->out, g_latest.p, g_latest.n);
            sb_puts(&c->out, "\n\n");
        }
        c->sse = true;
    } else respond(c, 404, "Not Found", "text/plain", nullptr, "not found\n", 10);
#undef ROUTE
done:
    if (head && !c->sse) {
        // Trim the body: everything after the header block.
        char *e = strstr(c->out.p, "\r\n\r\n");
        if (e) c->out.n = (size_t)(e + 4 - c->out.p);
    }
}

static int listen_on(const char *addr, int port) {
    int fd = -1;
    if (!addr || strchr(addr, ':')) {
        struct sockaddr_in6 a6 = {.sin6_family = AF_INET6, .sin6_port = htons((uint16_t)port)};
        if (addr && inet_pton(AF_INET6, addr, &a6.sin6_addr) != 1) return -1;
        if (!addr) a6.sin6_addr = in6addr_any;
        fd = socket(AF_INET6, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
        if (fd >= 0) {
            int on = 1, off = 0;
            setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);
            setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof off);
            if (!bind(fd, (struct sockaddr *)&a6, sizeof a6) && !listen(fd, 64)) return fd;
            close(fd);
        }
        if (addr) return -1;
    }
    // IPv4, or IPv6 is unavailable in this container.
    struct sockaddr_in a4 = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port)};
    if (addr && inet_pton(AF_INET, addr, &a4.sin_addr) != 1) return -1;
    fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (fd < 0) return -1;
    int on = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);
    if (bind(fd, (struct sockaddr *)&a4, sizeof a4) || listen(fd, 64)) {
        close(fd);
        return -1;
    }
    return fd;
}

static void tick(void) {
    sb row = {};
    g_latest.n = 0;
    emit_snapshot(&g_latest, &row);
    free(g_hist[g_hist_head]);
    g_hist[g_hist_head] = row.p; // ownership moves to the ring
    g_hist_head = (g_hist_head + 1) % HIST_LEN;
    if (g_hist_n < HIST_LEN) g_hist_n++;
    for (int i = 0; i < g_ncl; i++) {
        struct client *c = &g_cl[i];
        if (!c->sse) continue;
        if (c->out.n - c->off > MAX_SSE_BACKLOG) {
            cl_close(i--); // reader too slow; it will reconnect
            continue;
        }
        sb_puts(&c->out, "data: ");
        sb_put(&c->out, g_latest.p, g_latest.n);
        sb_puts(&c->out, "\n\n");
    }
}

static int serve(const char *addr, int port) {
    int lfd = listen_on(addr, port);
    if (lfd < 0) {
        fprintf(stderr, "nodestat: cannot listen on %s:%d: %s\n", addr ? addr : "*", port, strerror(errno));
        return 1;
    }
    fprintf(stderr, "nodestat: listening on %s:%d%s\n", addr ? addr : "*", port, g_token[0] ? " (token required)" : "");
    pthread_t th;
    pthread_create(&th, nullptr, host_thread, nullptr);
    pthread_create(&th, nullptr, gpu_thread, nullptr);
    sleep_s(0.3); // let the first samples land
    double next = now_mono();
    struct pollfd pf[MAXCLIENT + 1];
    for (;;) {
        double t = now_mono();
        if (t >= next) {
            tick();
            next += 1.0;
            if (next < t) next = t + 1.0;
        }
        pf[0] = (struct pollfd){.fd = lfd, .events = g_ncl < MAXCLIENT ? POLLIN : 0};
        for (int i = 0; i < g_ncl; i++)
            pf[i + 1] = (struct pollfd){.fd = g_cl[i].fd, .events = POLLIN | (g_cl[i].out.n > g_cl[i].off ? POLLOUT : 0)};
        int nfds = g_ncl + 1;
        int timeout = (int)ceil((next - now_mono()) * 1000);
        if (poll(pf, (nfds_t)nfds, timeout < 0 ? 0 : timeout) < 0 && errno != EINTR) return 1;

        if (pf[0].revents & POLLIN) {
            int fd;
            while (g_ncl < MAXCLIENT && (fd = accept4(lfd, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK)) >= 0)
                g_cl[g_ncl++] = (struct client){.fd = fd, .t0 = now_mono()};
        }
        // Walk backwards so cl_close's swap-with-last never skips a client.
        // Clients accepted just now have no pollfd entry yet; skip them.
        for (int i = nfds - 2; i >= 0; i--) {
            struct client *c = &g_cl[i];
            short re = pf[i + 1].revents;
            if (re & (POLLERR | POLLNVAL)) {
                cl_close(i);
                continue;
            }
            if (re & (POLLIN | POLLHUP)) {
                char tmp[4096];
                char *dst = c->sse ? tmp : c->in + c->inlen;
                size_t cap = c->sse ? sizeof tmp : sizeof c->in - 1 - c->inlen;
                ssize_t k = cap ? recv(c->fd, dst, cap, 0) : 0;
                if (k == 0 || (k < 0 && errno != EAGAIN && errno != EINTR)) {
                    cl_close(i);
                    continue;
                }
                if (k > 0 && !c->sse && !c->close_after) {
                    c->inlen += (size_t)k;
                    c->in[c->inlen] = 0;
                    if (strstr(c->in, "\r\n\r\n")) handle_request(c);
                    else if (c->inlen >= sizeof c->in - 1)
                        respond(c, 431, "Request Header Fields Too Large", "text/plain", nullptr, "", 0);
                }
            }
            if (c->out.n > c->off) {
                ssize_t k = send(c->fd, c->out.p + c->off, c->out.n - c->off, MSG_NOSIGNAL | MSG_DONTWAIT);
                if (k < 0 && errno != EAGAIN && errno != EINTR) {
                    cl_close(i);
                    continue;
                }
                if (k > 0) c->off += (size_t)k;
                if (c->off == c->out.n) c->off = c->out.n = 0;
            }
            if (c->close_after && c->out.n == 0) {
                cl_close(i);
                continue;
            }
            if (!c->sse && now_mono() - c->t0 > 15) cl_close(i);
        }
    }
}

// ------------------------------------------------------------------- main

static void usage(void) {
    fputs("usage: nodestat [-p PORT] [-b ADDR] [--www DIR]   serve the status page\n"
          "       nodestat --once                            print one JSON snapshot\n"
          "       nodestat --gpu-tag                         print e.g. 4x4070\n",
          stderr);
}

static void gpu_alarm(int sig) {
    (void)sig;
    static const char m[] = "nodestat: NVML did not answer within 10s\n";
    (void)!write(2, m, sizeof m - 1);
    _exit(2);
}

int main(int argc, char **argv) {
    int port = 8080;
    const char *addr = nullptr;
    bool once = false, tag = false;
    const char *ep = getenv("NODESTAT_PORT");
    if (ep && *ep) port = atoi(ep);
    const char *tok = getenv("NODESTAT_TOKEN");
    if (tok) copy_str(g_token, sizeof g_token, tok);
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) addr = argv[++i];
        else if (!strcmp(argv[i], "--www") && i + 1 < argc) g_www = argv[++i];
        else if (!strcmp(argv[i], "--once")) once = true;
        else if (!strcmp(argv[i], "--gpu-tag")) tag = true;
        else {
            usage();
            return !!strcmp(argv[i], "-h") && !!strcmp(argv[i], "--help");
        }
    }
    signal(SIGPIPE, SIG_IGN);
    host_static_init();
    if (tag || once) {
        signal(SIGALRM, gpu_alarm);
        alarm(10);
        struct cpu_prev pv = {.cg_usage_s = NAN};
        host_sample(&g_host, &pv);
        if (once) {
            sleep_s(0.5); // second sample so rates and percentages are defined
            host_sample(&g_host, &pv);
        }
        pthread_t th;
        pthread_create(&th, nullptr, gpu_thread, nullptr);
        for (int i = 0; i < 1000; i++) {
            pthread_mutex_lock(&g_gpu_mu);
            bool done = g_gpu.sweep_t > 0;
            pthread_mutex_unlock(&g_gpu_mu);
            if (done) break;
            sleep_s(0.01);
        }
        if (tag) {
            char out[160];
            gpu_tag(&g_gpu, out, sizeof out);
            puts(out);
        } else {
            sb b = {};
            emit_snapshot(&b, nullptr);
            puts(b.p);
        }
        return 0;
    }
    return serve(addr, port);
}
