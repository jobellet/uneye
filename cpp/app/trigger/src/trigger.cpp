// Trigger engine (task T19): rule matching on events, delayed firing on a dedicated timer thread, sinks, honest log of requested versus actual firing times.
// Host clock: eyelive/trigger_clock.hpp. `sample_arrival_host_s` of on_event is the host time at which the ONSET sample of the event arrived; for the Offset anchor the offset time is that value + (t_offset - t_onset) (tracker clock differences).
#include "eyelive/trigger.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <map>
#include <mutex>
#include <thread>
#include <chrono>
#include "eyelive/trigger_clock.hpp"
#include "sinks.hpp"

namespace eyelive {
namespace {


struct RuleSet { std::vector<TriggerRule> rules; std::vector<std::shared_ptr<Sink>> sinks; };

struct Pending {
    double due = 0, requested = 0, anchor_t = 0; float amp = 0; uint8_t cls = 0; int rule = 0; bool late = false; std::shared_ptr<const RuleSet> set;
};
struct Later { bool operator()(const Pending& a, const Pending& b) const { return a.due > b.due; } };

class TriggerEngine final : public ITriggerEngine {
 public:
    TriggerEngine() { pending_.reserve(4096); log_.reserve(1000); }
    ~TriggerEngine() override { stop(); }

    void set_rules(const std::vector<TriggerRule>& rules) override {
        auto set = std::make_shared<RuleSet>(); set->rules = rules; std::map<std::string, std::shared_ptr<Sink>> by_spec;
        { std::lock_guard<std::mutex> l(mu_); if (rules_) for (size_t i = 0; i < rules_->rules.size(); ++i) by_spec[rules_->rules[i].sink] = rules_->sinks[i]; }    // keep the sinks of unchanged specs (open sockets / files)
        for (const auto& r : rules) {
            auto it = by_spec.find(r.sink); if (it == by_spec.end()) { std::string err; auto s = make_sink(r.sink, &err); if (!s) { set_error(err); s = make_sink("null", nullptr); } it = by_spec.emplace(r.sink, s).first; }
            set->sinks.push_back(it->second);
        }
        if (running_) for (auto& s : set->sinks) { std::string err; if (!s->open(&err)) set_error(err); }
        std::lock_guard<std::mutex> l(mu_);
        std::vector<double> last(rules.size(), -1e300); if (rules_) for (size_t i = 0; i < rules.size(); ++i) for (size_t j = 0; j < rules_->rules.size(); ++j) if (rules_->rules[j].name == rules[i].name) last[i] = last_anchor_[j];
        last_anchor_ = std::move(last); rules_ = set;
    }

    void on_event(const Event& e, double arrival_s) override {
        std::lock_guard<std::mutex> l(mu_); if (!rules_ || !running_) return; const double now = trigger_host_now_s(); bool woke = false;
        const auto& rules = rules_->rules; const bool retraction = !e.provisional && e.confidence <= 0.0f;
        for (size_t i = 0; i < rules.size(); ++i) {
            const TriggerRule& r = rules[i]; if (!r.enabled) continue;
            if (!r.any_class && e.cls != r.cls) continue;
            if (retraction) continue;                                                           // a retracted provisional event cannot un-fire a trigger
            if (r.anchor == TriggerAnchor::OnsetProvisional) { if (!e.provisional) continue; } else if (e.provisional) continue;
            const bool constrained = r.amp_min_deg > 0.0f || r.amp_max_deg < 1e8f;
            if (constrained) { if (e.provisional) continue; if (e.amplitude_deg < r.amp_min_deg || e.amplitude_deg > r.amp_max_deg) continue; }    // provisional events carry no amplitude: a rule with an amplitude range needs a final anchor
            const double anchor = r.anchor == TriggerAnchor::Offset ? arrival_s + (e.t_offset - e.t_onset) : arrival_s;
            if (r.refractory_ms > 0 && anchor - last_anchor_[i] < r.refractory_ms / 1000.0) continue;
            last_anchor_[i] = anchor;
            Pending p; p.requested = anchor + r.delay_ms / 1000.0; p.late = p.requested < now; p.due = p.late ? now : p.requested; p.anchor_t = e.t_onset; p.amp = e.amplitude_deg; p.cls = e.cls; p.rule = int(i); p.set = rules_;
            if (pending_.size() == pending_.capacity()) { ++dropped_; continue; }                  // no allocation on the event path: a full queue drops (4096 pending triggers is far beyond any experiment)
            pending_.push_back(std::move(p)); std::push_heap(pending_.begin(), pending_.end(), Later()); woke = true;
        }
        if (woke) cv_.notify_one();
    }

    std::vector<TriggerLogEntry> log_snapshot() const override { std::lock_guard<std::mutex> l(log_mu_); std::vector<TriggerLogEntry> v; v.reserve(log_.size()); for (size_t i = 0; i < log_.size(); ++i) v.push_back(log_[(log_head_ + i) % log_.size()]); return v; }

    void start() override {
        std::lock_guard<std::mutex> lc(ctl_); if (running_) return;
        { std::lock_guard<std::mutex> l(mu_); if (rules_) for (auto& s : rules_->sinks) { std::string err; if (!s->open(&err)) set_error(err); } stop_ = false; running_ = true; }
        timer_ = std::thread([this] { loop(); });
    }
    void stop() override {
        std::lock_guard<std::mutex> lc(ctl_);
        { std::lock_guard<std::mutex> l(mu_); if (!running_) return; stop_ = true; } cv_.notify_all(); if (timer_.joinable()) timer_.join();
        std::lock_guard<std::mutex> l(mu_); running_ = false; pending_.clear();                    // triggers whose delay has not elapsed are dropped: an experiment that ends must not fire late triggers
    }
    std::string last_error() const { std::lock_guard<std::mutex> l(log_mu_); return error_; }

 private:
    void set_error(const std::string& e) { std::lock_guard<std::mutex> l(log_mu_); error_ = e; }
    static size_t fill_payload(char* out, size_t cap, const std::string& tpl, double t, float amp, int cls) {
        size_t n = 0; auto put = [&](const char* s, size_t len) { for (size_t i = 0; i < len && n + 1 < cap; ++i) out[n++] = s[i]; }; char num[48];
        for (size_t i = 0; i < tpl.size(); ++i) {
            if (tpl[i] == '{') { const size_t e = tpl.find('}', i); if (e != std::string::npos) { const std::string key = tpl.substr(i + 1, e - i - 1); int k = -1;
                if (key == "t") k = std::snprintf(num, sizeof num, "%.6f", t); else if (key == "amp") k = std::snprintf(num, sizeof num, "%.3f", double(amp)); else if (key == "cls") k = std::snprintf(num, sizeof num, "%d", cls);
                if (k >= 0) { put(num, size_t(k)); i = e; continue; } } }
            put(&tpl[i], 1);
        }
        out[n] = '\0'; return n;
    }
    void loop() {
        std::unique_lock<std::mutex> lk(mu_);
        while (!stop_) {
            if (pending_.empty()) { cv_.wait(lk); continue; }
            const double now = trigger_host_now_s(), due = pending_.front().due, wait = due - now, spin = spin_s_;
            if (wait > spin + 50e-6) {                                                            // sleep until `spin` before the due time (or until an earlier trigger / stop arrives), then spin; `spin` follows the measured oversleep of the OS timer
                const double target = now + wait - spin; cv_.wait_for(lk, std::chrono::duration<double>(wait - spin)); const double over = trigger_host_now_s() - target;
                if (over > 0) spin_s_ = std::min(0.004, std::max(spin_s_ * 0.98, 1.3 * over + 100e-6)); else spin_s_ = std::max(100e-6, spin_s_ * 0.98); continue;
            }
            if (wait > 0) { lk.unlock(); while (trigger_host_now_s() < due) {} lk.lock(); if (stop_) break; if (pending_.empty() || pending_.front().due > due) continue; }
            std::pop_heap(pending_.begin(), pending_.end(), Later()); Pending p = std::move(pending_.back()); pending_.pop_back(); lk.unlock();
            const TriggerRule& r = p.set->rules[size_t(p.rule)]; char buf[512]; const size_t n = fill_payload(buf, sizeof buf, r.payload, p.anchor_t, p.amp, p.cls);
            const double fired = trigger_host_now_s(); p.set->sinks[size_t(p.rule)]->send(buf, n);
            { std::lock_guard<std::mutex> l(log_mu_); TriggerLogEntry en{r.name, p.requested, fired, p.anchor_t, p.late}; if (log_.size() < 1000) log_.push_back(std::move(en)); else { log_[log_head_] = std::move(en); log_head_ = (log_head_ + 1) % 1000; } }
            lk.lock();
        }
    }

    mutable std::mutex mu_, log_mu_, ctl_; std::condition_variable cv_; std::shared_ptr<const RuleSet> rules_; std::vector<double> last_anchor_; std::vector<Pending> pending_;
    std::vector<TriggerLogEntry> log_; size_t log_head_ = 0; std::thread timer_; bool stop_ = false, running_ = false; double spin_s_ = 1.5e-3; uint64_t dropped_ = 0; std::string error_;
};

}  // namespace

std::unique_ptr<ITriggerEngine> make_trigger_engine() { return std::make_unique<TriggerEngine>(); }

}  // namespace eyelive
