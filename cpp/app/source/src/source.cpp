#include "eyelive/source.hpp"
#include "udp_source.hpp"
#include "file_source.hpp"
#include "eyelive/sim_source.hpp"

namespace eyelive {

std::unique_ptr<ISource> make_source(const SourceSpec& spec, std::string* error) {
    if (spec.kind == "udp") {
        auto src = std::make_unique<UdpSource>(spec);
        if (src->open(error)) return src;
        return nullptr;
    } else if (spec.kind == "file") {
        auto src = std::make_unique<FileSource>(spec);
        if (src->open(error)) return src;
        return nullptr;
    } else if (spec.kind == "sim") {
        auto src = std::make_unique<SimSource>(spec);
        if (src->open(error)) return src;
        return nullptr;
    } else {
        if (error) *error = "Unknown source kind: " + spec.kind;
        return nullptr;
    }
}

} // namespace eyelive
