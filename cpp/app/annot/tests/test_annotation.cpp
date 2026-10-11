#include "eyelive/annotation.hpp"
#ifdef NDEBUG
#undef NDEBUG      // the test relies on assert(): in Release it would be compiled out (and its parser, which advances inside assert(), would never terminate)
#endif
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

using namespace eyelive;

bool is_close(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

void run_property_test_with_render() {
    std::cout << "[Test] Property Test (10000 random edit sequences)\n";
    std::mt19937 gen(42);

    for (int run = 0; run < 100; ++run) {
        ExpertLabels labels(3);
        int64_t n_samples = 1000;

        std::vector<uint8_t> naive(n_samples, 0);

        for (int step = 0; step < 100; ++step) {
            int op = gen() % 10;
            switch(op) {
                case 0: {
                    int64_t s = gen() % (n_samples - 10);
                    int64_t e = s + (gen() % 50);
                    uint8_t c = (gen() % 2) + 1;
                    if (labels.add(s, e, c)) {
                        for (int64_t i = s; i <= std::min(e, n_samples - 1); ++i) {
                            naive[i] = c;
                        }
                    }
                    break;
                }
                case 1: {
                    if (!labels.events().empty()) {
                        size_t idx = gen() % labels.events().size();
                        labels.remove(idx);
                    }
                    break;
                }
                case 2: {
                    if (!labels.events().empty()) {
                        size_t idx = gen() % labels.events().size();
                        int64_t no = gen() % labels.events()[idx].offset;
                        labels.move_onset(idx, no);
                    }
                    break;
                }
                case 3: {
                    if (!labels.events().empty()) {
                        size_t idx = gen() % labels.events().size();
                        int64_t no = labels.events()[idx].onset + gen() % 50;
                        labels.move_offset(idx, no);
                    }
                    break;
                }
                case 4: {
                    if (!labels.events().empty()) {
                        size_t idx = gen() % labels.events().size();
                        uint8_t c = (gen() % 2) + 1;
                        labels.set_class(idx, c);
                    }
                    break;
                }
                case 5: {
                    if (!labels.events().empty()) {
                        size_t idx = gen() % labels.events().size();
                        const auto& ev = labels.events()[idx];
                        if (ev.offset > ev.onset + 1) {
                            int64_t at = ev.onset + 1 + gen() % (ev.offset - ev.onset);
                            labels.split(idx, at);
                        }
                    }
                    break;
                }
                case 6: {
                    if (labels.events().size() >= 2) {
                        size_t idx = gen() % (labels.events().size() - 1);
                        labels.merge(idx, idx + 1);
                    }
                    break;
                }
                case 7: {
                    labels.undo();
                    break;
                }
                case 8: {
                    labels.redo();
                    break;
                }
                case 9: {
                    std::vector<Event> pre(1);
                    pre[0].onset = gen() % (n_samples - 10);
                    pre[0].offset = pre[0].onset + (gen() % 50);
                    pre[0].cls = 1;
                    labels.load_prelabels(pre, 0, n_samples);
                    break;
                }
            }
        }

        const auto& evs = labels.events();
        for (size_t i = 0; i < evs.size(); ++i) {
            assert(evs[i].onset <= evs[i].offset);
            assert(evs[i].cls > 0 && evs[i].cls < labels.n_classes());
            if (i > 0) {
                assert(evs[i-1].offset < evs[i].onset);
            }
        }

        std::vector<uint8_t> my_naive(n_samples, 0);
        for (const auto& ev : evs) {
            int64_t s = std::max((int64_t)0, ev.onset);
            int64_t e = std::min(n_samples - 1, ev.offset);
            for (int64_t i = s; i <= e; ++i) {
                my_naive[i] = ev.cls;
            }
        }

        labels.exclude_range(100, 150, true);
        for(int64_t i=100; i<=150; ++i) my_naive[i] = kIgnoreLabel;

        std::vector<uint8_t> out;
        labels.render(0, n_samples - 1, false, out);

        assert(out.size() == my_naive.size());
        for (size_t i = 0; i < out.size(); ++i) {
            assert(out[i] == my_naive[i]);
        }
    }
    std::cout << "[Test] Property Test Passed\n";
}

void run_undo_redo_test() {
    std::cout << "[Test] Undo / Redo Test\n";
    ExpertLabels labels(3);
    labels.add(10, 20, 1);
    labels.add(30, 40, 2);
    assert(labels.events().size() == 2);

    labels.undo();
    assert(labels.events().size() == 1);

    labels.undo();
    assert(labels.events().empty());

    assert(!labels.undo());

    labels.redo();
    assert(labels.events().size() == 1);

    labels.redo();
    assert(labels.events().size() == 2);

    labels.clear_history();
    assert(labels.events().size() == 2);
    assert(!labels.undo());

    std::cout << "[Test] Undo / Redo Test Passed\n";
}

void run_save_load_test() {
    std::cout << "[Test] Save / Load Test\n";
    ExpertLabels labels1(3);
    labels1.add(10, 20, 1);
    labels1.exclude_range(50, 60, true);
    labels1.mark_reviewed(0, 100, true);

    std::string err;
    assert(labels1.save("test_save.json", &err));

    ExpertLabels labels2(3);
    assert(labels2.load("test_save.json", &err));

    assert(labels2.events().size() == 1);
    assert(labels2.events()[0].onset == 10);
    assert(labels2.events()[0].offset == 20);

    assert(labels2.excluded().size() == 1);
    assert(labels2.excluded()[0].first == 50);

    assert(labels2.reviewed_seconds(1.0) == 101.0);

    std::cout << "[Test] Save / Load Test Passed\n";
}

static void skip_ws(const std::string& json, size_t& pos) {
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t' || json[pos] == ',')) {
        pos++;
    }
}
static bool read_kw(const std::string& json, size_t& pos, const std::string& kw) {
    skip_ws(json, pos);
    if (json.compare(pos, kw.size(), kw) == 0) {
        pos += kw.size();
        return true;
    }
    return false;
}
static std::string read_str(const std::string& json, size_t& pos) {
    skip_ws(json, pos);
    if (pos >= json.size() || json[pos] != '"') return "";
    pos++;
    size_t end = pos;
    while (end < json.size() && json[end] != '"') end++;
    std::string res = json.substr(pos, end - pos);
    pos = end + 1;
    return res;
}
static double read_num(const std::string& json, size_t& pos) {
    skip_ws(json, pos);
    size_t end = pos;
    while (end < json.size() && (std::isdigit(json[end]) || json[end] == '.' || json[end] == '-' || json[end] == '+' || json[end] == 'e' || json[end] == 'E')) end++;
    std::string num = json.substr(pos, end - pos);
    pos = end;
    if (num.empty()) return 0;
    return std::stod(num);
}

void run_agreement_test(const std::string& golden_path) {
    std::cout << "[Test] Agreement Test\n";

    std::ifstream in(golden_path);
    assert(in.is_open());
    std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    size_t pos = 0;

    assert(read_kw(json, pos, "["));

    int cases_run = 0;

    while (pos < json.size() && json[pos] != ']') {
        skip_ws(json, pos);
        if (json[pos] == ']') break;
        assert(read_kw(json, pos, "{"));

        int64_t n_samples = 0;
        int tp=0, fp=0, fn=0;
        double f1=0.0, kappa=0.0;
        std::vector<Event> pred_ev;
        std::vector<Event> truth_ev;

        while (pos < json.size() && json[pos] != '}') {
            std::string key = read_str(json, pos);
            if (key.empty() && json[pos] == '}') break;
            assert(read_kw(json, pos, ":"));

            if (key == "n_samples") n_samples = (int64_t)read_num(json, pos);
            else if (key == "pred" || key == "truth") {
                std::vector<Event>* target = (key == "pred") ? &pred_ev : &truth_ev;
                assert(read_kw(json, pos, "["));
                while (pos < json.size() && json[pos] != ']') {
                    skip_ws(json, pos);
                    if (json[pos] == ']') break;
                    assert(read_kw(json, pos, "["));
                    int64_t s = (int64_t)read_num(json, pos);
                    int64_t e = (int64_t)read_num(json, pos);
                    assert(read_kw(json, pos, "]"));
                    Event ev; ev.onset = s; ev.offset = e; ev.cls = 1;
                    target->push_back(ev);
                }
                assert(read_kw(json, pos, "]"));
            } else if (key == "tp") tp = (int)read_num(json, pos);
            else if (key == "fp") fp = (int)read_num(json, pos);
            else if (key == "fn") fn = (int)read_num(json, pos);
            else if (key == "event_f1") f1 = read_num(json, pos);
            else if (key == "kappa") kappa = read_num(json, pos);
            else {
                // skip value
                skip_ws(json, pos);
                if (json[pos] == '"') read_str(json, pos);
                else read_num(json, pos);
            }
        }
        assert(read_kw(json, pos, "}"));

        AgreementStats st = agreement(pred_ev, truth_ev, n_samples, 1);

        if (st.fp != fp || st.tp != tp || st.fn != fn || !is_close(st.event_f1, f1, 1e-4) || !is_close(st.kappa, kappa, 1e-4)) {
             std::cout << "Mismatch in case " << cases_run << "\n";
             std::cout << "Expected: tp=" << tp << " fp=" << fp << " fn=" << fn << "\n";
             std::cout << "Actual:   tp=" << st.tp << " fp=" << st.fp << " fn=" << st.fn << "\n";
             assert(false);
        }
        cases_run++;
    }

    std::cout << "Cases run: " << cases_run << std::endl;
    assert(cases_run >= 30);
    std::cout << "[Test] Agreement Test Passed\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <cpp_dir>\n";
        return 1;
    }
    std::string cpp_dir = argv[1];
    std::string golden_path = cpp_dir + "/app/annot/tests/golden_agreement.txt";

    run_property_test_with_render();
    run_undo_redo_test();
    run_save_load_test();
    run_agreement_test(golden_path);

    return 0;
}
