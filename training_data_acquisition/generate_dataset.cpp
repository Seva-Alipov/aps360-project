#include "problems.hpp"

#include <curl/curl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cctype>

#include "problems.hpp"

namespace fs = std::filesystem;

// =========================
// Config for LM Studio
// =========================

// Your LM Studio endpoint & model from the curl you pasted
static const std::string LLM_URL   = "http://localhost:1234/v1/chat/completions";
static const std::string MODEL_NAME = "openai/gpt-oss-20b";

// System prompt: override the “Always answer in rhymes” with what we want:
static const std::string SYSTEM_PROMPT =
    "You are an expert C programmer. "
    "You receive algorithmic problem statements. "
    "You MUST respond with ONLY valid C source code for a single .c file. "
    "No markdown fences, your output must compile as is. "
    "Include a main() function that demonstrates or solves the problem.";

// Where to save _<problem_number>.c
static const fs::path OUTPUT_DIR = "../training_data";

// =========================
// Helpers
// =========================

// write callback
static size_t write_to_string(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t real_size = size * nmemb;
    std::string* mem = static_cast<std::string*>(userp);
    mem->append(static_cast<char*>(contents), real_size);
    return real_size;
}

// escape text for JSON string
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '\"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c; break;
        }
    }
    return out;
}

// pull out choices[0].message.content (works for LM Studio)
static std::string extract_content_from_response(const std::string& response) {
    // Find `"content"` key first
    const std::string key = "\"content\"";
    size_t pos = response.find(key);
    if (pos == std::string::npos) {
        // couldn't find content key; fallback to raw
        return response;
    }

    // Find the colon after "content"
    pos = response.find(':', pos + key.size());
    if (pos == std::string::npos) {
        return response;
    }
    ++pos; // move past ':'

    // Skip whitespace after colon: spaces, tabs, newlines, etc.
    while (pos < response.size() && std::isspace(static_cast<unsigned char>(response[pos]))) {
        ++pos;
    }

    // Next character should be starting quote
    if (pos >= response.size() || response[pos] != '\"') {
        // Unexpected format — fallback
        return response;
    }
    ++pos; // move past opening quote

    std::string result;
    bool escape = false;

    for (size_t i = pos; i < response.size(); ++i) {
        char c = response[i];

        if (escape) {
            // rudimentary unescape for common escapes
            switch (c) {
                case 'n': result += '\n'; break;
                case 'r': result += '\r'; break;
                case 't': result += '\t'; break;
                case '\\': result += '\\'; break;
                case '"': result += '"'; break;
                default: result += c; break;
            }
            escape = false;
        } else if (c == '\\') {
            escape = true;
        } else if (c == '"') {
            // end of the JSON string
            break;
        } else {
            result += c;
        }
    }

    return result;
}

// remove ``` fences if the model insists on them
static std::string strip_code_fences(std::string s) {
    if (s.rfind("```", 0) == 0) {
        auto pos = s.find('\n');
        if (pos != std::string::npos) {
            s = s.substr(pos + 1);
        } else {
            s.clear();
        }
    }
    auto last = s.rfind("```");
    if (last != std::string::npos) {
        s = s.substr(0, last);
    }
    return s;
}

static std::string clean_model_output(std::string s) {
    // If there are any weird tags before the real code,
    // jump to the first "#include" if present.
    std::size_t pos = s.find("#include");
    if (pos != std::string::npos) {
        s = s.substr(pos);
    }

    // If no includes, maybe the file starts with "int main"
    pos = s.find("int main");
    if (pos != std::string::npos && pos > 0 && s.find("#include") == std::string::npos) {
        s = s.substr(pos);
    }

    // Trim leading whitespace just in case
    while (!s.empty() && (s[0] == ' ' || s[0] == '\n' || s[0] == '\r' || s[0] == '\t')) {
        s.erase(s.begin());
    }

    return s;
}

// THIS is what main() should call: it returns ONLY the C code text
static std::string call_llm(const std::string& user_prompt) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "curl_easy_init failed\n";
        return {};
    }

    std::string response_data;

    // JSON payload for LM Studio
    std::string payload = "{"
        "\"model\":\"" + json_escape(MODEL_NAME) + "\","
        "\"messages\":["
            "{"
                "\"role\":\"system\","
                "\"content\":\"" + json_escape(SYSTEM_PROMPT) + "\""
            "},"
            "{"
                "\"role\":\"user\","
                "\"content\":\"" + json_escape(user_prompt) + "\""
            "}"
        "],"
        "\"temperature\":0.2,"
        "\"max_tokens\":-1,"
        "\"stream\":false"
    "}";

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, LLM_URL.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_string);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "curl_easy_perform failed: "
                  << curl_easy_strerror(res) << "\n";
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (response_data.empty()) return {};

    std::string content = extract_content_from_response(response_data);
    content = strip_code_fences(content);
    content = clean_model_output(content);
    return content;
}

// =========================
// Main: generate _<n>.c files
// =========================

int main() {
    fs::create_directories("../training_data");

    if (curl_global_init(CURL_GLOBAL_ALL) != 0) {
        std::cerr << "curl_global_init failed\n";
        return 1;
    }

    for (size_t i = 0; i < problems.size(); ++i) {
        int problem_number = static_cast<int>(i) + 1;
        fs::path out_path = "../training_data/_" + std::to_string(problem_number) + ".c";

        if (fs::exists(out_path)) {
            std::cout << "Skipping " << out_path << " (exists)\n";
            continue;
        }

        std::cout << "Generating problem " << problem_number << "...\n";

        std::string user_prompt =
            "Problem " + std::to_string(problem_number) + ":\n" +
            problems[i] +
            "\n\nWrite a single self-contained C program that solves this. "
            "Output ONLY C code.";

        // IMPORTANT: this is the parsed C code, not raw JSON
        std::string code = call_llm(user_prompt);

        if (code.empty()) {
            std::cerr << "Empty response for problem " << problem_number << "\n";
            continue;
        }

        std::ofstream out(out_path);
        if (!out) {
            std::cerr << "Failed to open " << out_path << "\n";
            continue;
        }

        out << code;
        out.close();

        std::cout << "Wrote " << out_path << "\n";
    }

    curl_global_cleanup();
    return 0;
}