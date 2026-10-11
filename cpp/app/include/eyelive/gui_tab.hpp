// GUI tab interface (CONTRACT). The shell (task T12) owns the window, the render loop and the tab registry; every tab (T13-T17) is one class in its own files and registers itself with a static registrar.
// Tabs only talk to IEngine. Dear ImGui is the only GUI dependency (cpp/third_party/imgui, fetched by cpp/third_party/fetch.sh).
#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "eyelive/engine.hpp"

namespace eyelive {

class ITab {
 public:
    virtual ~ITab() = default;
    virtual const char* title() const = 0;                       // text of the tab, e.g. "Acquire"
    virtual int order() const = 0;                               // position in the tab bar (Acquire 10, Live 20, Detection 30, Review 40, Train 50, Triggers 60, Predict 70, Log 80)
    virtual void draw(IEngine& engine) = 0;                      // called once per rendered frame inside the tab's ImGui window area
    virtual bool wants_redraw() const { return false; }          // true while the tab shows live data: the shell then keeps rendering at 30 Hz; otherwise it renders only on input events
};

using TabFactory = std::function<std::unique_ptr<ITab>()>;
std::vector<TabFactory>& tab_registry();                         // defined by the shell (T12): a function-local static vector
struct TabRegistrar { explicit TabRegistrar(TabFactory f) { tab_registry().push_back(std::move(f)); } };   // usage in a tab's .cpp: static eyelive::TabRegistrar reg([]{ return std::make_unique<AcquireTab>(); });

}  // namespace eyelive
