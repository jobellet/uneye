#include "eyelive/gui_tab.hpp"
namespace eyelive {
std::vector<TabFactory>& tab_registry() { static std::vector<TabFactory> r; return r; }
}
