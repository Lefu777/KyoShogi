#pragma once

#include <iostream>
#include <sstream>
#include <ctime>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <vector>

#include "config.hpp"
#include "cshogi.h"
#include "parallel_uct_node.hpp"
#include "base_player.hpp"
#include "util.hpp"
#include "nn_tensorrt.hpp"
#include "yobook_parser.hpp"
#include "stop_watch.hpp"
#include "peak_parser.hpp"
#include "dfpn_parallel_pv.hpp"
#include "my_dfpn_parallel5.hpp"
#include "types.hpp"

// TODO
//     : まずは、class のメンバ関数をthreadで並列実行して、
//       global変数 or static メンバ変数を正しくインクリメントする、というテストをしよう。
//       (staticメンバ変数は厳しそう & staticメンバ変数の使い方一通り調べたい & dlshogiはglobal変数使ってるっぽい。)

// HACK: これ、なんらかのnamespace に入れたいな、全体を。

//#ifdef PARALLEL_PUCT_PLAYER
#if 1

// プロトタイプ宣言的な？
class ParallelPuctSearcherGroup;
class ParallelPuctSearcher;


class ParallelDfpnGroup;
class ParallelPvDfpnGroup;

// HACK: _t に。
//////// typedef
typedef std::pair<UctNode*, int> UctNodeAndMove;
typedef std::pair<feature_t, Color> FeatureAndColor;
typedef std::pair<UctNode*, FeatureAndColor> UctNodeAndFeatureAndColor;
typedef std::vector<UctNodeAndMove> Trajectory;

// debug 用
inline std::string traj_to_str(const Trajectory& traj) {
	std::vector<std::string> traj_moves;
	for (const auto& uctnode_and_move : traj) {
		traj_moves.emplace_back(uctnode_and_move.first->child_infos[uctnode_and_move.second].move.toUSI());
	}
	return join(" ", traj_moves);
}

//////// extern static (キモ)

std::unique_ptr<UctTree>& get_global_tree();
// 一番探索すべき指し手のindex を返す。
// NOTE: 並列で呼ばれる
inline int select_max_ucb_child(uct_child_info_t* info, UctNode* node);

// NOTE
//     : ParallelPuctSearcherGroup のメンバ関数は並列で複数呼んではならない。
// dlshogi に倣ってGroup とした。やねうら王もこんな名前だったはず。
// 個人的には統括者、的な感じの方がしっくり来る気もするんだけど、まぁこういう話は二の次でいいや。
class ParallelPuctSearcherGroup {
private:
	NNTensorRT* _nn;
	std::vector<ParallelPuctSearcher> _searchers;
	std::vector<std::thread> _searcher_ths;
	std::mutex _gpu_mtx;

	// 子供たちに効力がある。
	std::atomic<bool> _shared_stop;
public:
	ParallelPuctSearcherGroup()
		: _nn(nullptr)
	{
	}

	~ParallelPuctSearcherGroup() {
		if (_nn != nullptr) {
			delete _nn;
		}
	}


	void stop() { _shared_stop.store(true); }
	bool _isready();

	void run();
	void join();


	bool nn_forward(const int& batch_size, const feature_t input, float* output_p, float* output_v);
};




// NOTE
//     : global変数は即ち、SearcherとPlayer が両方その情報が必要ということ。
//       (or 使うのはSearcherだけだが、ユーザーからの設定を一括で反映する為にglobal にする)
//     : (monitoring thread は全体で一つ。複数GPU であっても全体で一つ。
// TODO
//     : monitoring thread の実装をどうするか。
//     : _board をどうするか。頻繁にアクセスしてるようなら、コピーして各スレッドに渡してやるのが良いと思うけど。
//       -> _board が使われている箇所を確認。
//       -> should_exit_search() で.ply()
//          usi_loop() で.turn()
//          _isready() で.reset()
//          go() で、set_position_and_get_components
//         _go() で、global_tree->current_head->expand_node(_board);     // ルートノードの展開
//         _go() で、__LegalMoveList legal_moves(_board);    // 候補手が一つの場合は、その手を返す。
//             -> これならPlayer で持っててok
//         search() で、while(1) { __Board board_cpy(_board); }
//             -> これもするなら、global にしておいて、毎度searcher は探索開始時にそこから ParallelPuctSearcher::_board にcopy する。
class ParallelPuctSearcher {
private:
	//std::string _position_str;                                                                            //// Player
	//int last_pv_print_playout;                                                                       //// ===== global ?????　(monitoring threadの interrupution で必要。)
	//int time_limit;        // ms                                                                    //// ===== global ?????　(monitoring threadの interrupution で必要。)
	//int n_extend_time;     // 現在のroot局面において、何回探索延長をしたか                          //// ===== global ?????　(monitoring threadの interrupution で必要。)

	//YoBookParser _yobook_parser;                                                                       //// Player  (定跡管理はplayer だけで)
	//stopwatch_t stop_watch;                                                                       //// ===== global ????? (monitoring threadの print_bestmove で必要。)
	//go_info_t g_gi;                                                                                     //// Player

	//std::atomic<bool> is_stop;                                                                       //// ===== global =====
	//std::thread _searcher_thread;                                                                   //// Player
	//                                                                                                //// Player::goUct() { searcher_group.Run(); } なる goUct() を実行するスレッド。

	std::vector<UctNodeAndFeatureAndColor> _eval_queue;    // 推論待ちのqueue                         //// Searcher
	int _current_queue_idx;    // 一番初めは0. 現状のbatch_size とも言える。                          //// Searcher
	// nn
	ParallelPuctSearcherGroup* _parent_parallel_puct_searcher_group;
	feature_t _input;                                                                      //// Searcher
	float* _output_policy;                                                                 //// Searcher
	float* _output_value;                                                                 //// Searcher

	std::atomic<bool>* _shared_stop;

	bool _should_exit_search();

	void _cuda_free_input() {
#if defined (FEATURE_V2)
		cudaFree(std::get<0>(_input));
		cudaFree(std::get<1>(_input));
#else
		cudaFree(_input);
#endif
	}

public:
	ParallelPuctSearcher()
		: _current_queue_idx(0), _parent_parallel_puct_searcher_group(nullptr), _output_policy(nullptr), _output_value(nullptr)
	{
#if defined(FEATURE_V2)
		std::get<0>(_input) = nullptr;
		std::get<1>(_input) = nullptr;
#else
		_input = nullptr;
#endif
	}

	~ParallelPuctSearcher() {
		_cuda_free_input();
		cudaFree(_output_policy);
		cudaFree(_output_value);
	}

	// TODO: player の_isready で呼ばれる
	bool _isready(ParallelPuctSearcherGroup* parent_parallel_puct_searcher_group);

	// TODO: player の_usinewgame で呼ばれる
	void _usinewgame() const {
		;
	}

	void set_shared_stop(std::atomic<bool>* shared_stop) { _shared_stop = shared_stop; }
	void set_shared_stop_true() { _shared_stop->store(true); }

	// PUCT探索のエントリポイント
	// UCTSearcher::ParallelUctSearch()
	// NOTE: 並列で呼ばれる
	void parallel_search(const bool is_monitoring_thread);

	// https://qiita.com/amate/items/d1f84f6fbbf24e05e375
	// NOTE: 並列で呼ばれる
	// NOTE
	//     : result は、current_node の手番からみた結果である。
	float uct_search(__Board& board, uct_child_info_t* current_info, UctNode* current_node, Trajectory& trajectory);

	void update_result(UctNode* node, int next_index, float result) const;

    // NOTE
	//     : 実装上thread safe (あり得るのは後者のみ)
	//         : 同じオブジェクトから並列で呼ぶ場合、unsafe
	//         : 同じオブジェクトからは一つだけ、異なるオブジェクトから並列で呼ぶ場合、thread safe。
	// @arg node: 未評価のnode
	// @arg board: node に対応した局面
	void queue_node(UctNode* node, __Board& board);
	
	// queue の中身を一斉に評価
	// NOTE
	//     : 実装上thread safe のはず
	//         : 同じオブジェクトからは一つだけ、異なるオブジェクトから並列で呼ぶ場合、
	//           同じノードが異なるthread にqueue されていない(任意のnode のqueue される合計回数が1回である)
	//           ことが保証されるならば、thread safe。
	//             : node への書き込みが4件ある
	void eval_node();

};

// TODO
//     : 抽象クラスというか、基底クラスを作る。
//       ほんでもって、override 的なことして、基底のPlayerとしてオブジェクトは持ってるけど、
//       random_player のを呼び出す、的なのをしたい。
//     : なんか、header の中身変えただけだと、.exe に反映されん。
//     : TT はshared_ptr, group はunique_ptr 使ってやっても良さげ。
//       ただ、後者は楽だけど、前者はちょっとだけ面倒。

// NOTE
//     : ParallelPuctPlayer のメンバ関数は並列で複数呼んではならない。
class ParallelPuctPlayer : public BasePlayer {
private:
	std::mutex _position_str_mtx;
	std::string _position_str;
	YoBookParser _yobook_parser;

	std::unique_ptr<ParallelPuctSearcherGroup> _searcher_group;
	std::unique_ptr<ParallelDfpnGroup> _root_dfpn_group;
	std::unique_ptr<ParallelPvDfpnGroup> _pv_dfpn_group;
	std::unique_ptr<ParallelPvDfpnGroup> _pv_dfpn_group_2;

	ns_dfpn::TranspositionTable* _self_TT;
	ns_dfpn::TranspositionTable* _enem_TT;

	std::thread _go_thread;
	
	// 探索前に必ずfalse に。
	//std::atomic<bool> _shared_stop;    // HACK: global_stop の方がニュアンスとしては近そう。
	// NOTE: 
	std::atomic<bool> _go_stop;
	// NOTE: _stop_searcher() が呼ばれる時に、その先頭でtrue になる。
	//       GUI からのstop 以外のstop なら、join() し終わったあるgroup(root_dfpn or puct) が他を止めるので、
	//       join() 終わった時に既に_searcher_stop==true になっていれば、
	//       自分の都合ではなく他人の都合で止まったのだと分かる。
	//     : run() join() の1set が終わって、更にrun() する(止められないと止められない奴を実行する)なら、
	//       絶対にその前に_searcher_stop==false であることを確認する。
	std::atomic<bool> _searcher_stop;


	inline std::string _get_position_str_safe() {
		std::lock_guard<std::mutex> lock(_position_str_mtx);
		return _position_str;
	}

	inline void _set_position_str_safe(const std::string& position_str) {
		std::lock_guard<std::mutex> lock(_position_str_mtx);
		_position_str = position_str;
	}

	// run する前にtrue に。
	// 止めたい時にfalse に。
	//inline void _set_shared_stop_value(const bool x) { _shared_stop.store(x); }

	inline void _stop_and_join_go_if_running();
	inline void _stop_searcher();

	inline void _set_go_stop(const bool go_stop) { _go_stop.store(go_stop); }
	//inline void _set_shared_stop(const bool shared_stop) { _shared_stop.store(shared_stop); }

	inline [[carries_dependency]] bool _get_go_stop() const { return _go_stop.load(); }
	//inline [[carries_dependency]] bool _get_shared_stop() const { return _shared_stop.load(); }

public:
	ParallelPuctPlayer();
	~ParallelPuctPlayer();

	void usi_loop();
	void usi();

	// @arg name
	//     : setoption name <id> value <x>
	//       setoption は、usi_parser で取り除かれている
	void setoption(std::string cmd);

	// TODO: searcher の_isready() を呼ぶ。後、定跡とかの読み込みは変わらずこっちで。
	bool _isready();

	void isready();

	// もうこれ要らんやろ
	void usinewgame() {
		// 何もしないが形式上呼ばれる。
	}

	// @arg moves
	//     : 以下の2通りがある。
	//           : startpos moves xxxx xxxx ...
	//           : [sfen形式の任意の初期局面(先頭にsfenは無い)] moves xxxx xxxx ...
	virtual void position(const std::string& position_str);

	// https://github.com/TadaoYamaoka/python-dlshogi/blob/master/pydlshogi/player/mcts_player.py
	// 終局 or 宣言勝ち の局面では呼び出さない。
	std::pair<Move, Move> go_puct();

	std::pair<std::string, std::string> go_impl(const std::string& position_str);

	void go();
};

#endif    // PARALLEL_PUCT_PLAYER