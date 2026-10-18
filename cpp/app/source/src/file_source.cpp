#include "file_source.hpp"
#include <sstream>
#include <thread>
#include <chrono>

namespace eyelive {

FileSource::FileSource(const SourceSpec& spec) : spec_(spec) {}

FileSource::~FileSource() {
    close();
}

bool FileSource::open(std::string* error) {
    file_.open(spec_.path);
    if (!file_.is_open()) {
        if (error) *error = "Failed to open file: " + spec_.path;
        return false;
    }
    asc_.reset(); gap_total_s_ = 0; last_t_ = -1;
    { std::string head, l; for (int i = 0; i < 40 && std::getline(file_, l); ++i) head += l + "\n"; file_.clear(); file_.seekg(0, std::ios::beg);
      const std::string& p = spec_.path; std::string ext = p.size() >= 4 ? p.substr(p.size() - 4) : ""; for (char& c : ext) c = char(std::tolower(static_cast<unsigned char>(c)));
      if (ext == ".asc" || AscParser::looks_like_asc(head)) { asc_ = std::make_unique<AscParser>(spec_.eye); first_t_ = -1; finished_ = false; return true; } }
    std::string header;
    std::getline(file_, header);
    // basic check to see if it's a string header instead of a sample
    if (!header.empty() && !(header[0] == '-' || header[0] == '+' || header[0] == '.' || (header[0] >= '0' && header[0] <= '9'))) {
        // keep it skipped
    } else {
        // rewind if it was a data row
        file_.seekg(0, std::ios::beg);
    }
    finished_ = false;
    first_t_ = -1;
    return true;
}

void FileSource::close() {
    if (file_.is_open()) {
        file_.close();
    }
    finished_ = true;
}

std::string FileSource::description() const {
    return "file://" + spec_.path;
}

size_t FileSource::read(std::vector<Sample>& out, size_t max_samples, int timeout_ms) {
    if (finished_ || !file_.is_open()) return 0;
    (void)timeout_ms; // reading local file is usually fast enough not to block for timeout, we just read up to max_samples

    size_t initial_size = out.size();
    std::string line;

    while (out.size() - initial_size < max_samples && std::getline(file_, line)) {
        if (line.empty() || line[0] == '\r') continue;

        Sample s; bool got = false;
        if (asc_) { got = asc_->feed(line, s); if (!asc_->error().empty()) { finished_ = true; return out.size() - initial_size; } }
        else {
            // parse CSV replacing commas with spaces
            for (char& c : line) if (c == ',') c = ' ';
            std::stringstream ss(line);
            if (ss >> s.t >> s.x >> s.y) {
                uint32_t valid = 1;
                if (ss >> valid) {
                    if (!valid) s.flags = 0;
                }
                got = true;
            }
        }
        if (got) {
            if (spec_.pixels_per_degree > 0) {
                s.x /= spec_.pixels_per_degree;
                s.y /= spec_.pixels_per_degree;
            }

            if (first_t_ < 0) {
                first_t_ = s.t;
                start_time_ = std::chrono::steady_clock::now();
            } else if (spec_.speed > 0) {
                if (last_t_ >= 0 && s.t - last_t_ > 0.5) gap_total_s_ += s.t - last_t_ - 1.0 / 1000.0;           // a pause between two recording blocks: do not wait for it
                auto now = std::chrono::steady_clock::now();
                double elapsed_real_s = std::chrono::duration<double>(now - start_time_).count();
                double elapsed_sim_s = (s.t - first_t_ - gap_total_s_) / spec_.speed;
                if (elapsed_sim_s > elapsed_real_s) {
                    std::this_thread::sleep_for(std::chrono::duration<double>(elapsed_sim_s - elapsed_real_s));
                }
            }

            last_t_ = s.t; out.push_back(s);
        }
    }

    if (file_.eof()) {
        finished_ = true;
    }
    return out.size() - initial_size;
}

} // namespace eyelive
