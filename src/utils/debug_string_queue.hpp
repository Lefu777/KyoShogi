#pragma once

#include <vector>
#include <string>

extern std::vector<std::string> debug_str;

void enqueue_debug_str(const std::string& obj);
void enqueue_debug_str(std::string&& obj);
void flush_debug_str();
void clear_debug_str();
void flush_all_and_clear_debug_str();

void enqueue_inf_debug_str(const std::string& obj);
void enqueue_inf_debug_str(std::string&& obj);
void flush_inf_debug_str();

void debug_queue_str_init();

// NOTE
//     : https://learn.microsoft.com/ja-jp/cpp/error-messages/tool-errors/linker-tools-error-lnk2001?view=msvc-170
//       inline 関数の定義はヘッダーで。