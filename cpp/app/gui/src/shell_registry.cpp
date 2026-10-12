// GUI tab registry
#include "eyelive/gui_tab.hpp"

namespace eyelive {

std::vector<TabFactory>& tab_registry() {
    static std::vector<TabFactory> registry;
    return registry;
}

} // namespace eyelive
