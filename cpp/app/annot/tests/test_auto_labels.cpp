// apply_auto_labels: replaces automatic events in unreviewed parts only, never touches expert events or reviewed ranges, and is not an undo step (Undo neither removes nor resurrects automatic labels).
#include <cstdio>
#include <vector>
#include "eyelive/annotation.hpp"
using namespace eyelive;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
static Event ev(int64_t a, int64_t b, uint8_t cls = 1) { Event e; e.onset = a; e.offset = b; e.cls = cls; e.origin = EventOrigin::Detector; return e; }
int main() {
    ExpertLabels ex(3);
    ex.apply_auto_labels({ev(100, 130), ev(500, 540), ev(900, 940)}, 0, 1000); CHECK(ex.events().size() == 3 && ex.prelabels().size() == 3, "initial auto labels: %zu / %zu", ex.events().size(), ex.prelabels().size());
    CHECK(!ex.undo(), "automatic labels must not create an undo step");
    // the expert deletes the event at 500 and its neighbourhood becomes reviewed (silently: one Undo reverts both)
    CHECK(ex.remove(1), "remove"); ex.mark_reviewed(400, 650, true, false); CHECK(ex.events().size() == 2, "after delete");
    // new automatic labels for the whole range: the reviewed part (400-650) is untouched, the rest is replaced
    ex.apply_auto_labels({ev(105, 128), ev(510, 530), ev(905, 935)}, 0, 1000);
    bool has_510 = false, has_105 = false, has_905 = false; for (auto& e : ex.events()) { has_510 |= e.onset == 510; has_105 |= e.onset == 105; has_905 |= e.onset == 905; }
    CHECK(!has_510, "a deleted event came back inside a reviewed range"); CHECK(has_105 && has_905, "unreviewed parts were not replaced"); CHECK(ex.events().size() == 2, "events: %zu", ex.events().size());
    // an expert event is never replaced
    CHECK(ex.add(200, 220, 1), "add"); ex.apply_auto_labels({ev(210, 240), ev(300, 320)}, 150, 350); bool kept = false, blocked = true; for (auto& e : ex.events()) { kept |= e.onset == 200 && e.origin == EventOrigin::Expert; blocked &= e.onset != 210; }
    CHECK(kept && blocked, "expert event replaced or overlapped by an automatic one");
    // Undo reverts the expert's add, and the automatic labels stay as they are (the old snapshots were updated too)
    CHECK(ex.undo(), "undo"); bool still_300 = false, no_200 = true; for (auto& e : ex.events()) { still_300 |= e.onset == 300; no_200 &= e.onset != 200; }
    CHECK(still_300 && no_200, "undo of the add changed the automatic labels"); CHECK(ex.undo(), "second undo (delete + its range)"); bool back_500 = false; for (auto& e : ex.events()) back_500 |= (e.onset == 500 || e.onset == 510); CHECK(back_500, "undo of the delete did not bring the event back"); CHECK(ex.reviewed().empty(), "undo must also remove the silent reviewed range");
    std::printf(fails ? "FAILED\n" : "auto label tests passed\n"); return fails ? 1 : 0;
}
