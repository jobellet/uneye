// EyeLink ASC reading: synthetic file written from the documented format (not a real EyeLink recording).
#include <eyelive/source.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
using namespace eyelive;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::cout << "FAIL line " << __LINE__ << ": " #c "\n"; ++fails; } } while (0)

static std::vector<Sample> read_all(const std::string& path, const std::string& eye, std::string* err) {
    SourceSpec sp; sp.kind = "file"; sp.path = path; sp.speed = 0; sp.eye = eye;
    auto src = make_source(sp, err); std::vector<Sample> all; if (!src) return all;
    for (int i = 0; i < 100; ++i) { std::vector<Sample> v; if (src->read(v, 1000, 20) == 0 && i > 3) break; all.insert(all.end(), v.begin(), v.end()); }
    return all;
}

int main() {
    const std::string p1 = "test_asc_mono.asc", p2 = "test_asc_bino.asc", p3 = "test_asc_href.asc";
    { std::ofstream f(p1);
      f << "** CONVERTED FROM: test.edf\n** EYELINK II 1\nMSG\t1000 !CAL\nSTART\t1000 \tRIGHT\tSAMPLES\tEVENTS\nPRESCALER\t1\nVPRESCALER\t1\nPUPIL\tAREA\nEVENTS\tGAZE\tRIGHT\tRATE\t 500.00\tTRACKING\tCR\tFILTER\t2\nSAMPLES\tGAZE\tRIGHT\tRATE\t 500.00\tTRACKING\tCR\tFILTER\t2\tINPUT\n";
      f << "1000\t  512.0\t  384.0\t  900.0\t...\n1002\t  514.0\t  384.5\t  901.0\t...\n1004\t   .\t   .\t    0.0\t...\n1006\t  520.0\t  390.0\t  899.0\t...\n";
      f << "SFIX R 1000\nEFIX R 1000 1004 4 513 384 900\nSSACC R 1006\nESACC R 1006 1020 14 520.0 390.0 600.0 400.0 5.20 380\nMSG 1010 trial 1\nEND\t1030 \tSAMPLES\tEVENTS\tRES\t 40.0\t 40.0\n"
        << "START\t5000 \tRIGHT\tSAMPLES\tEVENTS\nSAMPLES\tGAZE\tRIGHT\tRATE\t 500.00\n5000\t 100.0\t 100.0\t 800.0\t...\n"; }
    { std::ofstream f(p2);
      f << "SAMPLES\tGAZE\tLEFT\tRIGHT\tRATE\t1000.00\tTRACKING\tCR\n10\t100.0\t200.0\t800.0\t102.0\t202.0\t810.0\t...\n11\t.\t.\t0.0\t104.0\t204.0\t810.0\t...\n"; }
    { std::ofstream f(p3); f << "SAMPLES\tHREF\tRIGHT\tRATE\t1000.00\n10\t1.0\t2.0\t800\n"; }

    std::string err; auto a = read_all(p1, "auto", &err);
    CHECK(a.size() == 5);
    if (a.size() == 5) {
        CHECK(std::fabs(a[0].t - 1.000) < 1e-9 && std::fabs(a[1].t - 1.002) < 1e-9);
        CHECK(std::fabs(a[0].x - 512.f) < 1e-3 && std::fabs(a[1].y - 384.5f) < 1e-3);
        CHECK(!(a[2].flags & kFlagValid));                                  // '.' = missing
        CHECK(a[3].flags & kFlagValid);
        CHECK(std::fabs(a[4].t - 5.0) < 1e-9);                              // second recording block
    }
    auto b = read_all(p2, "auto", &err); CHECK(b.size() == 2);
    if (b.size() == 2) { CHECK(std::fabs(b[0].x - 101.f) < 1e-3 && std::fabs(b[0].y - 201.f) < 1e-3);            // mean of both eyes
                         CHECK(std::fabs(b[1].x - 104.f) < 1e-3); CHECK(b[1].flags & kFlagValid); }                // left missing: right only
    auto l = read_all(p2, "left", &err); CHECK(l.size() == 2); if (l.size() == 2) { CHECK(std::fabs(l[0].x - 100.f) < 1e-3); CHECK(!(l[1].flags & kFlagValid)); }
    auto r = read_all(p2, "right", &err); CHECK(r.size() == 2); if (r.size() == 2) CHECK(std::fabs(r[0].x - 102.f) < 1e-3);
    auto h = read_all(p3, "auto", &err); CHECK(h.empty());
    std::remove(p1.c_str()); std::remove(p2.c_str()); std::remove(p3.c_str());
    std::cout << (fails ? "FAILED\n" : "OK\n"); return fails ? 1 : 0;
}
