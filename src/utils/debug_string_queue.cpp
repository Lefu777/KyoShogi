#include "debug_string_queue.hpp"

#include <iostream>
#include <atomic>
#include <sstream>
#include <mutex>
#include "util.hpp"

// HACK: めっちゃ適当に書いてるので綺麗にしたい。。。

// queue と言いつつ、速度を落とさないためにvector を使ってる。(atomic<int> でindex 制御するだけで良いので)
// queue だと、それ自体をlock しないといけない気がして、一旦vector にしてる。
// (dfpn の並列のバグ、速度出てた方が起きやすいと思ったので。)

constexpr int DEBUG_STR_SIZE = 80000;
std::vector<std::string> debug_str(DEBUG_STR_SIZE, "");
static std::atomic<int> debug_str_idx = 0;    // まだ書き込んでない先頭のidx
static int flush_start_idx = 0;    // まだ表示してない奴の先頭のidx
std::mutex flush_mtx;

// スレッドセーフ
void enqueue_debug_str(const std::string& obj) {
	const auto& idx = debug_str_idx++;
	if (idx < DEBUG_STR_SIZE) {
		debug_str[idx] = obj;
	}
}

// スレッドセーフ
void enqueue_debug_str(std::string&& obj) {
	const auto& idx = debug_str_idx++;
	if (idx < DEBUG_STR_SIZE) {
		debug_str[idx] = std::move(obj);
	}
}

// flush と言いつつ、これまでに使った領域はもう使えるようにならない。
// mod DEBUG_STR_SIZE をidx とすればどんどん上書きしていけるけど、それはそれで微妙な気もしなくはない。
// スレッドセーフ
void flush_debug_str() {
	flush_mtx.lock();
	const int crrt_idx = debug_str_idx;
	const int lim = my_min(DEBUG_STR_SIZE, crrt_idx);
	for (int i = flush_start_idx; i < lim; ++i) {
		//std::cout << "[" << i << "] : " << debug_str[i] << std::endl;
		std::cout << debug_str[i];
	}
	std::cout << "\nlim = " << lim << std::endl;
	flush_start_idx = crrt_idx;
	flush_mtx.unlock();
}

// enqueue, flush 系全てとデータ競合起こす(debug_str へのwrite) ので、シリアルに動かさなければならない。
// clear して、再度初めからstart
void clear_debug_str() {
	debug_str.clear();
	debug_str_idx = 0;
}

// enqueue_debug_str の呼び出しが完全に終わった後に呼び出すことが保証されている前提。
// queue の中身を先頭(index==0) から最後まで全部表示して、要素全部clear
void flush_all_and_clear_debug_str() {
	//// print
	const int crrt_idx = debug_str_idx;
	const int lim = my_min(DEBUG_STR_SIZE, crrt_idx);
	for (int i = 0; i < lim; ++i) {
		std::cout << "[" << i << "] : " << debug_str[i] << std::endl;
		//std::cout << debug_str[i];
	}
	std::cout << "\nlim = " << lim << std::endl;

	//// clear
	clear_debug_str();
}


//// 以下、無限にenqueue 出来る奴

// 無限にenqueue 出来る
// スレッドセーフ
void enqueue_inf_debug_str(const std::string& obj) {
	const auto&& idx = debug_str_idx++;
	debug_str[idx % DEBUG_STR_SIZE] = obj;
}

// 無限にenqueue 出来る
// スレッドセーフ
void enqueue_inf_debug_str(std::string&& obj) {
	const auto&& idx = debug_str_idx++;
	debug_str[idx % DEBUG_STR_SIZE] = std::move(obj);
}

// enqueue_inf_debug_str() に対応するflush
// スレッドセーフ
void flush_inf_debug_str() {
	flush_mtx.lock();
	const int crrt_idx = debug_str_idx;
	if (crrt_idx >= DEBUG_STR_SIZE) {
		// 上書きされてるなら全部出力
		const int lim = debug_str_idx + DEBUG_STR_SIZE;
		std::cout << "\n[flush all start]" << std::endl;
		for (int i = debug_str_idx; i < lim; ++i) {
			std::cout << debug_str[i % DEBUG_STR_SIZE];
		}
		std::cout << "\n[flush all done]" << std::endl;
	}
	else {
		// 上書きされていないなら、前回の続きからスタート
		std::cout << "\n[lim = " << crrt_idx << ":start]" << std::endl;
		for (int i = flush_start_idx; i < crrt_idx; ++i) {
			std::cout << debug_str[i];
		}
		flush_start_idx = crrt_idx;
		std::cout << "\n[lim = " << crrt_idx << ":done]" << std::endl;
	}

	flush_mtx.unlock();
}

void debug_queue_str_init() {
	debug_str.reserve(DEBUG_STR_SIZE);
}
