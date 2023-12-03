#pragma once

#include <atomic>
#include <thread>
#include <chrono>
#include <vector>
#include <memory>
#include <unordered_set>
#include "cshogi.h"
#include "parallel_uct_node.hpp"
#include "parallel_puct_player.hpp"

#ifdef DFPN_PARALLEL4
#include "my_dfpn_parallel4.hpp"
#elif defined DFPN_PARALLEL5
#include "my_dfpn_parallel5.hpp"
#else
static_assert("any dfpn is not available");
#endif

// TODO
//     : ひとまず、とにもかくにもIsWin とかを安全に出来る実装に。
//     : 置換表を一つこちらで用意して、各dfpn にポインタをset するように変更しよう。
//     : 個の実装、全部my_dfpn_parallel3.hpp にまとめて書いてもええかもな。
//     : threadid の偶奇とかに応じて、一番目の候補を選ぶとか、二番目の候補選ぶとか。。。
//           : この辺り、例の論文に何か書いてそう。
//     : 後は、dynamic widening とか。。。(候補手多い場合に有効だけど。。。)
// NOTE
//     : 置換表を共有するなら、任意の局面において、任意のスレッドが保持するboard のgamePly() が一致しなければならない。
//     : Lazy SMP なので、思考した後も、その局面が思考済みであることを覚える必要は無くて、
//       純粋に置換表を共有したら後は脳死探索してもらうのが基本スタイル。
//       ただ、同時に同じ局面を思考するのは効率が悪いので、そこはやめて欲しい。
class ParallelDfpnThread {
private:
	std::atomic<bool> _is_done;    // 結果を格納

	std::thread _th;

	//__Board* _root_board;
	__Board _root_board;
	int64_t _n_searched_nodes;
	int _this_threadid;    // _threadid は予約語
	ParallelDfPn _dfpn;
public:

	//// TODO: 再帰的に呼ぶ
	//void _search_with_nn(__Board& board) {
	//	;
	//}

	//void search_with_nn() {
	//	while (!_stop) {
	//		__Board board_cpy(*_root_board);
	//		_search_with_nn(board_cpy);
	//	}
	//}

	//// TODO: 再帰的に呼ぶ
	//bool _search(__Board& board) {
	//	_n_searched_nodes = 0;
	//	return _dfpn.dfpn<true>(board.pos, _n_searched_nodes, _this_threadid);
	//}

	//bool search() {
	//	__Board board_cpy(*_root_board);
	//	return _search(board_cpy);
	//}

	// 対局開始前の初期化
	void init(int max_depth, uint64_t max_node, int draw_ply) {
		this->set_maxdepth(max_depth).set_max_search_node(max_node).set_draw_ply(draw_ply);
	}

	void run() {
		// https://cpprefjp.github.io/reference/thread/thread/op_assign.html
		//     : 移動代入
		try {
			// NOTE
			//      : 実行する奴のstop フラグは、スレッド実行の手前でやるべし。
			_dfpn.init_result();
			_dfpn.dfpn_stop(false);
			_n_searched_nodes = 0;
    		_th = std::thread([&]() {
    			_dfpn.dfpn<false>(_root_board.pos, _n_searched_nodes, _this_threadid);
    			//sync_cout << "info: _this_threadid = " << _this_threadid << ", _is_mate = " << bts(_is_mate) << sync_endl;
    			//_is_done.store(true, std::memory_order_release);
    		});
		}
		catch (...) {
			std::cout << "Error!!!: at run(), got error, threadid = " << _this_threadid << std::endl;
			throw;
		}
	}

	void join() {
		_th.join();
	}

	void stop() {
		_dfpn.dfpn_stop(true);
	}

	// run より前に必ずset すること。
	void set_root_board(const __Board& root_board) {
		_root_board = root_board;
		//std::cout << "info: set_root_board() " << &(_root_board.pos) << std::endl;
	}

	// run より前に必ずset すること。
	ParallelDfpnThread& set_tt(ns_dfpn::TranspositionTable* x) {
		_dfpn.set_tt(x);
		return *this;
	}

	// run より前に必ずset すること。
	void set_threadid(const int threadid) {
		_this_threadid = threadid;
	}

	// join() 後に呼ぶこと
	Move get_move(__Board& board) { return _dfpn.dfpn_move(board.pos); }
	int get_pn() const { return _dfpn.get_result_pn(); }
	int get_dn() const { return _dfpn.get_result_dn(); }
	int64_t get_searched_node() const { return _n_searched_nodes; }
    std::tuple<std::string, int, Move> get_pv(__Board& board) { return _dfpn.get_pv<true>(board.pos); }

	// 以下3つはどれもデータ競合を起こさない
	bool get_is_mate() const { return _dfpn.get_result_is_mate(); }
	bool get_is_repeat() const { return _dfpn.get_result_is_repeat(); }
	bool get_is_done() const { return _dfpn.get_result_is_done(); }

	void print_addr() const {
		_root_board.pos.print_addr();
	}

private:

	//// TParallelDfPn のsettar のwrapper

	ParallelDfpnThread& set_maxdepth(int x) {
		_dfpn.set_maxdepth(x);
		return *this;
	}

	ParallelDfpnThread& set_max_search_node(uint64_t x) {
		_dfpn.set_max_search_node(x);
		return *this;
	}
	
	ParallelDfpnThread& set_draw_ply(int x) {
		_dfpn.set_draw_ply(x);
		return *this;
	}

	ParallelDfpnThread& set_shared_stop(std::atomic<bool>* x) {
		_dfpn.set_shared_stop(x);
		return *this;
	}

	ParallelDfpnThread& set_shared_stop_value(bool x) {
		_dfpn.set_shared_stop_value(x);
		return *this;
	}
};


// 一つの局面について、並列で詰み探索を行う際に用いる。
// PPDG.set_root_board(board).set_hash_size(32768).set_n_threads(16)
class ParallelDfpnGroup {
private:
	// TODO
	//     : ParallelPvDfpn を管理。
	//     : ここでtt をset
	//     : ここで設定をset
	//     : ここで、dfpn_parallel_pv.cpp 内のstatic なglobal_board にroot_board をコピー
	ns_dfpn::TranspositionTable* _TT;
	__Board _root_borad;
	int _n_threads;
	std::vector<std::unique_ptr<ParallelDfpnThread>> _dfpns;

	// 結果を格納
	int _result_pn;
	int _result_dn;

	std::atomic<bool> _stop;             // 内部だけに通用する
public:
	ParallelDfpnGroup()
		: _TT(nullptr), _n_threads(-1), _result_pn(-1), _result_dn(-1), _stop(false)
	{
	}

	// 異なるスレッド数でinit() した後は、必ず一度はset_tt() しなければならない。
	// 対局開始前の初期化
	void init(int n_threads, int max_depth, uint64_t max_node, int draw_ply) {
		//// init parent
		this->set_and_alloc_threads_if_changed(n_threads);
		//// init child
		this->__set_threadid();
		this->__init(max_depth, max_node, draw_ply);
	
		//this->__set_shared_stop();
		sync_cout
			<< "info string ParallelDfpnGroup:"
			<< ",_n_threads=" << _n_threads<< sync_endl;
	}

	void run() {
		_stop.store(false);
		_result_pn = -1;
		_result_dn = -1;

		for (int i = 0; i < _n_threads; ++i) {
			_dfpns[i]->run();
		}
	}

	void stop() {
		_stop.store(true);
		this->__stop();
	}

	// HACK: join() で結果返すのきめぇ～
	void join() {
		if (_n_threads > 0) {
			bool is_mate = false;
			bool is_any_done = false;
			uint64_t wait_loop_count = 0;

			while (true) {
				++wait_loop_count;
				for (int i = 0; i < _n_threads; ++i) {
					is_mate |= _dfpns[i]->get_is_mate();
					is_any_done |= _dfpns[i]->get_is_done();    // 1個が終わったら、他も終わってよい。
					if (is_any_done) {
						break;
					}
				}
				// まだ他の奴は探索続けてるかもしれないからstop させる。
				if (is_any_done) {
					//this->set_shared_stop_value(true);

					this->__stop();
					break;
				}
				std::this_thread::sleep_for(std::chrono::microseconds(25));
			}

			// 結果が出て全部にstop を送ったら実行。
			// set_stop(), join(), どちらも非constメンバ関数 なので書き込み扱いなんじゃないかな～。ちょっと分かってない。
			for (int i = 0; i < _n_threads; ++i) {
				_dfpns[i]->join();
			}

			_result_pn = _dfpns[0]->get_pn();
			_result_dn = _dfpns[0]->get_dn();
		}
		else {
			// しょうみ、pn が0 じゃなかったら、なんでもよい。
			_result_pn = INT_MAX;
			_result_dn = INT_MAX;
		}
	}

	// 毎局面ごとにset しなければならない。
	// 自分にも子供にもset する。
	void set_root_board(const __Board& root_board) {
		_root_borad = root_board;
		for (int i = 0; i < _n_threads; ++i) {
			_dfpns[i]->set_root_board(_root_borad);
		}
	}
	void set_tt(ns_dfpn::TranspositionTable* TT) {
		_TT = TT;
		this->__set_tt();
	}

	//// getter
	Move get_move(__Board& board) {
		if (_n_threads > 0) {
			return _dfpns[0]->get_move(board);
		}
		else {
			return Move::moveNone();
		}
	}
	int get_pn() const { return _result_pn; }
	int get_dn() const { return _result_dn; }
	bool get_is_mate() const { return _result_pn == 0; }

	int64_t get_searched_node() const {
		uint64_t retVal = 0;
		for (auto&& dfpn : _dfpns) {
			retVal += dfpn->get_searched_node();
		}
		return retVal;
	}

	uint16_t get_generation() const {
		return _TT->generation;
	}

	std::tuple<std::string, int, Move> get_pv(__Board& board) {
		if (_n_threads > 0) {
			return _dfpns[0]->get_pv(board);
		}
		else {
			return {"", -1, Move::moveNone() };
		}
	}

private:
	//// 管理するスレッド達
	// 対局前に一度set すれば問題ない
	// スレッド数をset して、スレッドを確保。
	void set_and_alloc_threads_if_changed(int n_threads) {
		if (_n_threads != n_threads) {
			_n_threads = n_threads;
			_dfpns.resize(_n_threads);
			for (auto&& dfpn : _dfpns) {
				dfpn = std::make_unique<ParallelDfpnThread>();
			}
		}
	}

	//// 子供たちに影響
	// 子供たち全体をstop 出来る。
	//void __set_shared_stop_value(bool stop) {
	//	_shared_stop = stop;
	//}

	// 探索終了時に呼び出す。
	ParallelDfpnGroup& __stop() {
		//std::cout
		//	<< "       ";
		for (int i = 0; i < _n_threads; ++i) {
			//std::cout << " " << i;
			_dfpns[i]->stop();
		}
		//std::cout << std::endl;
		return *this;
	}


	// 対局前に一度set すれば問題ない
	// set_n_threads の後に行うこと。
	ParallelDfpnGroup& __init(int max_depth, uint64_t max_node, int draw_ply) {
		for (auto&& dfpn : _dfpns) {
			dfpn->init(max_depth, max_node, draw_ply);
		}
		return *this;
	}

	// 対局前に一度set すれば問題ない
	// set_n_threads の後に行うこと。
	ParallelDfpnGroup& __set_tt() {
		if (_TT == nullptr) {
			sync_cout << "info string Error!: ParallelDfpnGroup, _TT is nullptr." << std::flush << sync_endl;
			throw std::runtime_error("_TT is nullptr!");
		}

		for (auto&& dfpn : _dfpns) {
			dfpn->set_tt(_TT);
		}
		return *this;
	}

	// 対局前に一度set すれば問題ない
	// set_n_threads の後に行うこと。
	ParallelDfpnGroup& __set_threadid() {
		for (int i = 0; i < _n_threads; ++i) {
			_dfpns[i]->set_threadid(i);
		}
		return *this;
	}

	//// 対局前に一度set すれば問題ない
	//// set_n_threads の後に行うこと。
	//ParallelDfpnGroup& __set_draw_ply(int x) {
	//	for (auto&& dfpn : _dfpns) {
	//		dfpn->set_draw_ply(x);
	//	}
	//	return *this;
	//}

	//// 対局前に一度set すれば問題ない
	//// set_n_threads の後に行うこと。
	//ParallelDfpnGroup& __set_maxdepth(int x) {
	//	for (auto&& dfpn : _dfpns) {
	//		dfpn->set_maxdepth(x);
	//	}
	//	return *this;
	//}

	//// 対局前に一度set すれば問題ない
	//// set_n_threads の後に行うこと。
	//ParallelDfpnGroup& __set_max_search_node(uint64_t x) {
	//	for (auto&& dfpn : _dfpns) {
	//		dfpn->set_max_search_node(x);
	//	}
	//	return *this;
	//}

	//ParallelDfpnGroup& __set_shared_stop() {
	//	for (auto&& dfpn : _dfpns) {
	//		dfpn->set_shared_stop(&_shared_stop);
	//	}
	//	return *this;
	//}
};


// TODO
//     : ParallelPvDfpnGroup という名前は不適切かも。ParallelDfpnGroup にしよう。
//     : ParallelDfpnGroup と同じ構造で、新たにParallelPvDfpnGroup を作ろうかね？
//     : やりたいことは、root で置換表A を共有して16 スレッドで詰み探索しつつ、
//       PV上を、置換表Aを共有した6スレッドと、置換表Bを共有した6スレッドを走らせたい。
//       -> いや、pv上の全ての局面を、2スレッドずつで探索とかの方が良いのでは？
//          2スレッドで、2倍の探索効率が出るようにすればよい。
//       -> PV を上から順に辿っていき、未探索局面に到達し次第、is_mate = ParallelDfpnGroup::run() みたいなことをして、
//          もしis_mate == true ならSetWin() する。
//          後、これに伴い、PUCT でもAnd/Or 木 チックに、子ノードのWin, Lose を元に親ノードのWin, Lose をset.
//     : NewSearch() は、puct のroot が1手進むごとに1つ進めれば良いはず。
//     : ParallelDfpnGroup も、_shared_stop へのポインタを持つように。。
//     : shared_stop の方が都合が良いので、ParallelPvDfpn をshared_stop に戻す。そして、それで念のためテストする。
//     : メンバは、
//       //// 現状はNewSearch() すれば実質的な置換表のclear なので以下の実装で良いが、世代変わっても再利用するなら、先手後手で分けるべき。
//       //// ただ、GHI問題が起きても次の世代に影響しないという利点もあり、世代変わっても一旦は再利用しない。
//       //// 二つのhash_size はそれぞれユーザー側で設定できるようにする。
//       //// 目安は、丁度スレッド数の比と一致するようにすべきだが、2**n にしないといけないという制約があるので、
//       //// "32GBと 16GB" とかが無難と思われる。メモリ不足を怖がらないなら、"64GBと 16GB" とかも良いが、グレー。
//       // root 側の手番の置換表。
//       _self_TT
//       _self_hash_size_mb
//       // root と反対の手番の置換表。
//       _enem_TT
//       _enem_hash_size_mb
//       
//       // これで全てを統括。
//       _shared_stop
//       // 
//       //// スレッド数 (root をN スレッド、PV上をM スレッドで探索する場合)
//       // root を探索するスレッド数。
//       _n_root_threads
//       // 先後合わせて、_n_pv_threads * 2 のスレッドが動く。
//       _n_pv_threads
//       //// スレッド数 (PV上のノードをN スレッド並列で解いていく場合)
//       // 使用可能な合計スレッド数。
//       // ex> 
//       // _n_threads = 28, _n_threads_per_node = 2 なら、self側が2スレ*7組, enem側が2スレ*7組 の計28組が動く。
//       _n_threads
//       // 1node 当たりのスレッド数。
//       _n_threads_per_node

// TODO
//     : ParallelPvDfpnThread を作成して、
//       そこでwhile 文で永遠にPV上を彷徨わせる。
//       そいつを、ParallelPvDfpnGroup で統合。


class ParallelPvDfpnThread {
private:
	std::atomic<bool> _stop;            // run をstop
	std::atomic<bool>* _shared_stop;    // 外部からのstop (使わない方針に変更。)
	std::thread _th;

	__Board _root_board;
	ParallelDfpnGroup _dfpn_group;

	ns_dfpn::TranspositionTable* _self_TT;
	ns_dfpn::TranspositionTable* _enem_TT;

	std::unordered_set<uct_child_info_t*>* _shared_searched_nodes;    // HACK: 正直pos::getKey() がkey でも良いのでは。
	std::mutex* _shared_searched_nodes_mtx;

	inline [[carries_dependency]] bool is_stop() const {
		if (_stop) {
			return true;
		}
		else if (_shared_stop != nullptr) {
			if (_shared_stop->load()) {
				return true;
			}
		}
		return false;
	}

public:
	ParallelPvDfpnThread() 
		: _stop(false), _shared_stop(nullptr), _self_TT(nullptr), _enem_TT(nullptr),
	      _shared_searched_nodes(nullptr), _shared_searched_nodes_mtx(nullptr)
	{}

	// 対局開始前の初期化
	void init(
		int n_threads, int max_depth, uint64_t max_node, int draw_ply,
		std::unordered_set<uct_child_info_t*>* shared_searched_nodes, std::mutex* shared_searched_nodes_mtx
	) {
		_dfpn_group.init(n_threads, max_depth, max_node, draw_ply);
		_shared_searched_nodes = shared_searched_nodes;
		_shared_searched_nodes_mtx = shared_searched_nodes_mtx;
	}

	void stop() {
		_stop = true;
		_dfpn_group.stop();
	}

	// node の最善の子ノードにどんどん再帰的に遷移していく。
	// 未探索局面に到達したら詰み探索を実行して終了。
	template<bool root_is_self>
	void _run(__Board& board, UctNode* node);

	void run();

	void join() {
		_th.join();
	}

	void set_root_board(const __Board& board) { _root_board = board; }

	void set_tt(ns_dfpn::TranspositionTable* self_TT, ns_dfpn::TranspositionTable* enem_TT) {
		_self_TT = self_TT;
		_enem_TT = enem_TT;
	}

	// _shared_stop を使わない方針に切り替えた。
	// (nullptr なら何も怒らないようになっているので、この関数が無ければ_shared_stop は何も出来まい。)
	//void set_shared_stop(std::atomic<bool>* shared_stop) {
	//	_shared_stop = shared_stop;
	//	_dfpn_group.set_shared_stop(shared_stop);
	//}
};

// 一つの局面について、並列で詰み探索を行う際に用いる。
// PPDG.set_root_board(board).set_hash_size(32768).set_n_threads(16)
class ParallelPvDfpnGroup {
private:
	// TODO
	//     : ParallelPvDfpn を管理。
	//     : ここでtt をset
	//     : ここで設定をset
	//     : ここで、dfpn_parallel_pv.cpp 内のstatic なglobal_board にroot_board をコピー
	ns_dfpn::TranspositionTable* _self_TT;    // self が勝ちか否かを探索(OrNode の手番==self)。
	ns_dfpn::TranspositionTable* _enem_TT;    // enem が勝ちが否かを探索(OrNode の手番==enem)。
	//int64_t _self_hash_size_mb;
	//int64_t _enem_hash_size_mb;

	int _draw_ply;
	int _max_depth;
	int _max_search_node;

	__Board _root_borad;
	int _n_threads;    // ParallelPvDfpnThread の数
	int _n_threads_per_node;    // ParallelPvDfpnThread 当たりのスレッド数
	std::vector<std::unique_ptr<ParallelPvDfpnThread>> _pv_dfpns;
	std::unordered_set<uct_child_info_t*> _shared_searched_nodes;
	std::mutex _shared_searched_nodes_mtx;
public:
	ParallelPvDfpnGroup()
		: _self_TT(nullptr), _enem_TT(nullptr),
		  /*_self_hash_size_mb(-1), _enem_hash_size_mb(-1), */_n_threads(-1), _n_threads_per_node(-1)
	{}

	// 対局前に呼ぶ想定
	// @arg n_threads
	//     : 合計で使用できるスレッド数
	//     : _n_threads = total_threads / n_threads_per_node
	// @arg n_threads_per_node: ParallelPvDfpnThread 当たりのスレッド数
	void init(
		ns_dfpn::TranspositionTable* self_TT, ns_dfpn::TranspositionTable* enem_TT,
		int total_threads, int n_threads_per_node, /*int self_hash_size_mb, int enem_hash_size_mb,*/
		int max_depth, uint64_t max_node, int draw_ply
	) {
		// TODO
		//     : 実装中。
		//// init parent
		this->set_TT(self_TT, enem_TT);
		this->set_and_alloc_threads_if_changed(total_threads / n_threads_per_node, n_threads_per_node);
		_shared_searched_nodes.clear();
		//// init child
		this->__set_tt();
		this->__init(max_depth, max_node, draw_ply);

		sync_cout
			<< "info string ParallelPvDfpnGroup:"
			<< ",total=" << total_threads
			<< ",_n_threads=" << _n_threads
			<< ",n_threads_per_node=" << n_threads_per_node << sync_endl;
	}

	~ParallelPvDfpnGroup() {
		//delete _self_TT;
		//delete _enem_TT;
	}

	void stop() {
		for (auto&& pv_dfpn : _pv_dfpns) {
			pv_dfpn->stop();
		}
	}

	// 事前にtt_new_search() を呼ぶこと。
	void run() {
		for (auto&& pv_dfpn : _pv_dfpns) {
			pv_dfpn->run();
		}
	}

	void join() {
		for (auto&& pv_dfpn : _pv_dfpns) {
			pv_dfpn->join();
		}
	}

	// 新しい探索の前には必ず一度呼ぶこと。
	//void tt_new_search() {
	//	_self_TT->NewSearch();
	//	_enem_TT->NewSearch();
	//}

	// 使わない方針に変更。
	//void set_shared_stop(std::atomic<bool>* shared_stop) {
	//	for (auto&& pv_dfpn : _pv_dfpns) {
	//		pv_dfpn->set_shared_stop(shared_stop);
	//	}
	//}

	// NOTE
	//     : 一応、子供にコピーをあげて終わる。関数を抜けたらその瞬間全ての処理が終わる。
	void set_root_board(const __Board& board) {
		_root_borad = board;
		for (auto&& pv_dfpn : _pv_dfpns) {
			pv_dfpn->set_root_board(_root_borad);
		}
	}

	void set_TT(ns_dfpn::TranspositionTable* self_TT, ns_dfpn::TranspositionTable* enem_TT) {
		_self_TT = self_TT;
		_enem_TT = enem_TT;
	}

	//ns_dfpn::TranspositionTable* get_self_TT() const { return _self_TT; }
	//ns_dfpn::TranspositionTable* get_enem_TT() const { return _enem_TT; }


	//// 毎局面ごとにset しなければならない。
	//// こいつは、自分にも子供にもset する。
	//void set_root_board(const __Board& root_board) {
	//	_root_borad = root_board;
	//	for (int i = 0; i < _n_threads; ++i) {
	//		_dfpn_groups[i]->set_root_board(_root_borad);
	//	}
	//}

private:
	//int64_t get_searched_node() const {
	//	uint64_t retVal = 0;
	//	for (auto&& dfpn : _dfpns) {
	//		retVal += dfpn->get_searched_node();
	//	}
	//	return retVal;
	//}


	//// 置換表
	// 対局前に一度set すれば問題ない
	//ParallelPvDfpnGroup& set_self_hash_size(int hash_size_mb) {
	//	_self_hash_size_mb = hash_size_mb;
	//	return *this;
	//}

	//ParallelPvDfpnGroup& set_enem_hash_size(int hash_size_mb) {
	//	_enem_hash_size_mb = hash_size_mb;
	//	return *this;
	//}
	
	// 対局前に一度alloc すれば問題ない
	//void alloc_tt() {
	//	_self_TT->Resize(_self_hash_size_mb);
	//	_enem_TT->Resize(_enem_hash_size_mb);
	//}

	void __set_tt() const {
		for (auto&& pv_dfpn : _pv_dfpns) {
			pv_dfpn->set_tt(_self_TT, _enem_TT);
		}
	}

	//// スレッド
	// 子ノードへのset はこれが終わった後に行うこと！
	void set_and_alloc_threads_if_changed(int n_threads, int n_threads_per_node) {
		if (_n_threads != n_threads) {
			_n_threads = n_threads;
			_pv_dfpns.resize(_n_threads);
			for (auto&& pv_dfpn : _pv_dfpns) {
				pv_dfpn = std::make_unique<ParallelPvDfpnThread>();
			}
		}
		// total_threads, n_threads_per_node が同時に変わると、_n_threads は変化しない場合があるので、
		// _n_threads が変化してない時でも代入する。
		_n_threads_per_node = n_threads_per_node;
	}

	// set_n_threads の後に行うこと。
	void __init(int max_depth, uint64_t max_node, int draw_ply) {
		for (auto&& pv_dfpn : _pv_dfpns) {
			pv_dfpn->init(
				_n_threads_per_node, max_depth, max_node, draw_ply, &_shared_searched_nodes, &_shared_searched_nodes_mtx
			);
		}
	}

	//ParallelDfpnGroupGroup& set_stop(bool stop) {
	//	//std::cout
	//	//	<< "       ";
	//	for (int i = 0; i < _n_threads; ++i) {
	//		//std::cout << " " << i;
	//		_dfpn_groups[i]->set_stop(stop);
	//	}
	//	//std::cout << std::endl;
	//	return *this;
	//}



	////// TParallelDfPn のsettar のwrapper
	//// 対局前に一度set すれば問題ない
	//// set_n_threads の後に行うこと。
	//ParallelDfpnGroupGroup& set_draw_ply(int x) {
	//	for (auto&& dfpn_group : _dfpn_groups) {
	//		dfpn_group->set_draw_ply(x);
	//	}
	//	return *this;
	//}

	//// 対局前に一度set すれば問題ない
	//// set_n_threads の後に行うこと。
	//ParallelDfpnGroupGroup& set_maxdepth(int x) {
	//	for (auto&& dfpn_group : _dfpn_groups) {
	//		dfpn_group->set_maxdepth(x);
	//	}
	//	return *this;
	//}

	//// 対局前に一度set すれば問題ない
	//// set_n_threads の後に行うこと。
	//ParallelDfpnGroupGroup& set_max_search_node(int x) {
	//	for (auto&& dfpn_group : _dfpn_groups) {
	//		dfpn_group->set_max_search_node(x);
	//	}
	//	return *this;
	//}

	//// 対局前に一度set すれば問題ない
	//// set_n_threads の後に行うこと。
	//ParallelDfpnGroupGroup& __set_tt() {
	//	for (auto&& dfpn_group : _dfpn_groups) {
	//		dfpn_group->set_tt(_TT);
	//	}
	//	return *this;
	//}
};