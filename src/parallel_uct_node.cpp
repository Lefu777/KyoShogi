#include "parallel_uct_node.hpp"

//#if defined(PARALLEL_PUCT_PLAYER)
#if 1

// https://learn.microsoft.com/ja-jp/cpp/error-messages/tool-errors/linker-tools-error-lnk2005?view=msvc-170
//std::unique_ptr<UctTree> global_tree;

constexpr int GC_INTERVAL_MS = 250;

static class UctNodeGC {
private:
	std::vector<std::unique_ptr<UctNode>> _gc_queue;
	std::mutex _queue_mtx;
	std::atomic<bool> _stop;
	std::thread _gc_thread;

	void _gc() {
		//std::cout << "info string _gc_queue.size() = " << _gc_queue.size() << std::endl;

		while (!_stop.load()) {
			// NOTE: コイツはlock の後に解放される。lock の最中には解放されない & 解放されるようにしてはいけない。
			std::unique_ptr<UctNode> destructed;

			if (!_gc_queue.empty()) {
				{
					// NOTE: あくまでlock が必要なのはデータ競合を起こしうる_gc_queue のread とwrite 操作!
					std::lock_guard<std::mutex> lock(_queue_mtx);
					destructed = std::move(_gc_queue.back());
					_gc_queue.pop_back();    // 抜け殻を削除
				}
			}
			else {
				break;
			}
		}
	}

public:
	UctNodeGC() {
		_stop.store(false);
		_gc_thread = std::thread([this]() {
			    this->run();
			}
		);
	}

	// .exe が落ちると同時にGC 終了
	~UctNodeGC() {
		_stop.store(true);
		_gc_thread.join();
	}

	void queue_node_to_gc(std::unique_ptr<UctNode> node) {
		if (node) {
			// debug
			std::ostringstream oss;
			stopwatch_t __sw1;
			stopwatch_t __sw2;
			stopwatch_t __sw3;

			__sw1.start();
			_queue_mtx.lock();
			__sw1.stop();

			__sw2.start();
			_gc_queue.emplace_back(std::move(node));    // ここでGC 側に所有権が移動
			__sw2.stop();
			
			__sw3.start();
			_queue_mtx.unlock();
			__sw3.stop();

			// debug
			oss
				<< "queue_node_to_gc{time1=" << __sw1.elapsed_ms() << ",time2=" << __sw2.elapsed_ms() << ",time3=" << __sw3.elapsed_ms() << "}";
			enqueue_debug_str(oss.str());
		}
	}

	void run() {
		while (!_stop.load()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(GC_INTERVAL_MS));
			_gc();
		}
	}
} uct_node_gc;


// @arg move: この指し手に対応する子ノード以外を全て解放する。
UctNode* UctNode::release_children_except_one(Move move) {
	// debug
	std::ostringstream oss;
	stopwatch_t __sw1;
	stopwatch_t __sw2;    // loop 全体を計測
	stopwatch_t __sw3;    // 発見した場合を計測
	std::vector<std::string> gced_infos;

	__sw2.start();
	if (legal_moves_size > 0) { gced_infos.reserve(legal_moves_size); }

	// FIXME: legal_moves_size > 0 では？
	if (legal_moves_size >= 0) {
		bool found = false;
		// HACK: 参照渡しした方が速い。
		for (int i = 0; i < legal_moves_size; ++i) {
			// NOTE
			//     : dlshogi だとis_lose, is_draw, is_win の時、move の上位3bit にそいつをset してるので、
			//       指し手は一致しても、move(の値) 自体は一致しないので再利用されない。
			//       ただ、これで再利用できてしまうと、特に千日手の場合に未評価の状態の奴がroot となってしまい、
			//       そいつには推論結果が無いのでucb が計算出来ず、select_max_ucb() がバグってしまう。(1番初めの指し手を無条件に返すようになってしまう。)
			//       なので、それを防ぐ為の仕様という可能性もあるので、バグと断定することは出来ない。。。
			if (child_infos[i].move == move) {    // 発見。
				__sw3.start();
				found = true;
				if (!child_nodes[i]) {    // 実体が無ければ作る。
					//std::cout << "info string rceo(): meet nullptr start" << std::endl;
					create_child_node(i);
					//child_nodes[i] = std::make_unique<UctNode>();
					//std::cout << "info string rceo(): meet nullptr done" << std::endl;
				}
				// else { // 実体がすでに有れば何もせずにそれを再利用。 }

				// 残すnode を0番目に持ってくる
				if (i != 0) {    // 0番目でない時
					// NOTE
					//     : 他の要素は入れ替えてないけど、使わんやろ
					//    -> 後そもそも、0番目のポインタは解放済みなので、このポインタは保持しなくても良い。
					child_nodes[0] = std::move(child_nodes[i]);
					child_infos[0] = child_infos[i];
				}
				__sw3.stop();
				std::ostringstream oss_found;
				oss_found << "found_idx=" << i;
				gced_infos.emplace_back(oss_found.str());
			}
			else {
				//std::cout << "info string rceo(): got = " << __move_to_usi(child_moves[i]) << std::endl;
				if (child_nodes[i]) {
					// debug
					stopwatch_t __sw4;
					__sw4.start();
					std::ostringstream oss_gced;
					oss_gced << "{i=" << i << ",move=" << child_infos[i].move.toUSI() << ",count=" << child_nodes[i]->move_count ;

					// GC
					uct_node_gc.queue_node_to_gc(std::move(child_nodes[i]));

					// debug
					__sw4.stop();
					oss_gced << ",time=" << __sw4.elapsed_ms() << "}";
					gced_infos.emplace_back(oss_gced.str());
				}
			}
		}
		__sw2.stop();

		// debug
		if (gced_infos.size() > 0) {
			oss
				<< "move=" << move.toUSI()
				<< ", legal_moves_size=" << legal_moves_size
				<< ", release{time2="
				<< __sw2.elapsed_ms() << ",time3=" << __sw3.elapsed_ms()
				<< "},\ngceds={" << join(",", gced_infos) << "}";
		}
		else {
			oss
				<< "move=" << move.toUSI()
				<< ", legal_moves_size=" << legal_moves_size
				<< ", release{time2="
				<< __sw2.elapsed_ms() << ",time3=" << __sw3.elapsed_ms()
				<< "}";
		}
		enqueue_debug_str(oss.str());

		if (found) {
			// 探索ではもうこのnode には触れない"はず"なので、legal_moves_size にwrite してもデータ競合は起きない。
			legal_moves_size = 1;
			return child_nodes[0].get();
		}
		else {
			init_and_create_single_child_node(move);
			// 実体を作成して、それを返す。
			// HACK: return create_child_node(0);
			return create_child_node(0);
			//return (child_nodes[0] = std::make_unique<UctNode>()).get();
		}
	}
	else {
		__sw1.start();
		init_and_create_single_child_node(move);
		__sw1.stop();

		// debug
		oss
			<< "move=" << move.toUSI()
			<< ", legal_moves_size=" << legal_moves_size
			<< ", release{time1=" << __sw1.elapsed_ms() << "}";
		enqueue_debug_str(oss.str());

		// 実体を作成して、それを返す。
		return create_child_node(0);
		//return (child_nodes[0] = std::make_unique<UctNode>()).get();
	}
}

void UctTree::_reset_and_set_to_current(std::unique_ptr<UctNode>& node) {
	if (node) {
		uct_node_gc.queue_node_to_gc(std::move(node));
	}
	// 探索木のroot node なのでcreate_child_node() を介さずに確保してよい。
	node = std::make_unique<UctNode>();
	current_head = node.get();
}


#endif