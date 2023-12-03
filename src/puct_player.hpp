#pragma once

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <thread>
#include <atomic>
#include <vector>

#include "config.hpp"
#include "cshogi.h"
#include "uct_node.hpp"
#include "base_player.hpp"
#include "util.hpp"
#include "nn_tensorrt.hpp"
#include "yobook_parser.hpp"
#include "stop_watch.hpp"
#include "peak_parser.hpp"

// TODO
//     : まずは、class のメンバ関数をthreadで並列実行して、
//       global変数 or static メンバ変数を正しくインクリメントする、というテストをしよう。
//       (staticメンバ変数は厳しそう & staticメンバ変数の使い方一通り調べたい & dlshogiはglobal変数使ってるっぽい。)

//extern std::unique_ptr<UctTree> global_tree;
//extern UctTree* global_tree;

//std::unique_ptr<UctTree> global_tree;


typedef std::pair<UctNode*, int> UctNodeAndMove;
typedef std::pair<float*, Color> FeatureAndColor;
typedef std::pair<UctNode*, FeatureAndColor> UctNodeAndFeatureAndColor;
typedef std::vector<UctNodeAndMove> Trajectory;

// 並列探索時の仮想loss
constexpr int VIRTUAL_LOSS = 1;

// ALL
//     : 以下は、探索時のresult の特別な場合。
//     : 値に意味はなく、playout or NN により得られるvalue の範囲(0 <= value <= 1) 以外ならok.
//     : 報酬としては用いられない(不適)。
// VALUE_XXX
//     : NNの推論よりも圧倒的に信頼度の高いもの(というか絶対的な評価)として、node.value に代入される。
// RESULT_XXX
//     : uct_search() の戻り値に、報酬(として用いれない値)が返る時に用いられる。
constexpr int VALUE_WIN = 40000;
constexpr int VALUE_DRAW = 20000;
constexpr int VALUE_LOSE = 10000;
constexpr float RESULT_QUEUING = -1;
constexpr float RESULT_DISCARDED = -2;

// 探索パラメータ
constexpr int puct_c_base = 19652;
constexpr float puct_c_init = 1.25;

// HACK; 50, 25 ぐらいの方が良いと思うけど、どうやろうか。
// 持ち時間制御に使用
constexpr int div_init = 30;    // 初期値(最大値)
//constexpr int div_second = 50;
constexpr int div_final = 14;    // 最小値
//constexpr int n_self_ply_while_begining = (div_init - div_second) / 2;    // 序盤に自分が指す手の数


// TODO
//     : 抽象クラスというか、基底クラスを作る。
//       ほんでもって、override 的なことして、基底のPlayerとしてオブジェクトは持ってるけど、
//       random_player のを呼び出す、的なのをしたい。
//     : なんか、header の中身変えただけだと、.exe に反映されん。

class PuctPlayer : public BasePlayer {
private:
	__Board _board;

	std::string _position_str;
	unsigned long long _playout_count;
	//UctTree _tree;
	//UctTree* global_tree;
	//static UctTree* global_tree;
	std::vector<UctNodeAndFeatureAndColor> _eval_queue;    // 推論待ちのqueue
	int _current_queue_idx;    // 一番初めは0. 現状のbatch_size とも言える。
	int _last_pv_print_playout;
	int _time_limit;        // ms
	int _n_extend_time;     // 現在のroot局面において、何回探索延長をしたか

	YoBookParser yobook_parser;
	stopwatch_t _stop_watch;
	go_info_t _gi;

	std::atomic<bool> _is_stop;
	std::thread _searcher_thread;

	// nn
	NNTensorRT* _nn;
	float* _input;
	float* _output_policy;
	float* _output_value;
	
	// NOTE: _const_playout が有効(1以上)であれば、これが一番強い優先順位を持つ。
	// USI option
	int _n_threads;
	int _batch_size;
	int _minimum_thinking_time;
	int _network_delay;    // 通信遅延を考慮して、これだけ余裕を持つ。
	int _const_playout;
	int _print_pv_per_playout;
	std::string _onnx_model;
	std::string _book_filename;
	int _book_moves;    // 何手目まで定跡を用いるか

	// 探索停止のフラグを立てる
	void _set_uct_search_stop() {
		_is_stop = true;
	}

	// 探索停止のフラグを取り下げる
	void _set_uct_search_start() {
		_is_stop = false;
	}

	void _stop_searcher_thread_if_running() {
		_set_uct_search_stop();
		// https://cpprefjp.github.io/reference/thread/thread/joinable.html
		if (_searcher_thread.joinable()) {
			_searcher_thread.join();
		}
	}

public:
	PuctPlayer()
		: BasePlayer(), _nn(nullptr), _onnx_model("model.onnx"), _book_filename("standard_book.db")
	{
		_board = __Board();
		//global_tree.reset(new UctTree());
		global_tree = std::make_unique<UctTree>();
		//global_tree = new UctTree();
		_position_str = "";
		_playout_count = 0;
		_current_queue_idx = 0;
		_last_pv_print_playout = 0;
		_time_limit = 0;
		_n_extend_time = 0;

		_is_stop = false;

		_n_threads = 2;
		_batch_size = 128;    // TODO: default option をどこかに。。。
		_minimum_thinking_time = 2000;
		_network_delay = 1000;
		_const_playout = 100000;
		_print_pv_per_playout = 10000;    // TODO: 時間制御諸々
		_book_moves = 16;

	}

	void usi_loop() {
		while (true) {
			std::string cmd, token;
			getline(std::cin, cmd);
			const auto&& tokens = tokenize(cmd);
			if (tokens.size() > 0) {
				token = tokens[0];
			}
			else {
				token = "";
			}

			if (token == "usi") {
				usi();
			}
			else if (token == "setoption") {
				setoption(cmd.substr(10));
			}
			else if (token == "isready") {
				isready();
			}
			else if (token == "usinewgame") {
				usinewgame();
			}
			else if (token == "position") {
				position(cmd.substr(9));
			}
			else if (token == "quit") {
				_stop_searcher_thread_if_running();
				exit(1);
			}
			else if (token == "stop") {
				_stop_searcher_thread_if_running();
			}
			else if (token == "gameover") {
				_stop_searcher_thread_if_running();
			}
			else if (token == "go") {
				// https://zenn.dev/yohhoy/articles/quiz-init-vector
				const std::vector tokens_without_go(tokens.begin() + 1, tokens.end());

				peak_parser_t parser(tokens_without_go);
				_gi.reset();

				while (parser.peak_next_if_exists()) {
					const auto& pt = parser.get_peak_token();
					if (pt == "infinite") {
						_gi.is_infinite = true;
					}
					else if (pt == "btime") {
						parser.peak_next_expected_to_exist();
						parser.store_peak_token(_gi.time[Black]);
					}
					else if (pt == "wtime") {
						parser.peak_next_expected_to_exist();
						parser.store_peak_token(_gi.time[White]);
					}
					else if (pt == "binc") {
						parser.peak_next_expected_to_exist();
						parser.store_peak_token(_gi.inc[Black]);
					}
					else if (pt == "winc") {
						parser.peak_next_expected_to_exist();
						parser.store_peak_token(_gi.inc[White] );
					}
					else if (pt == "byoyomi") {
						parser.peak_next_expected_to_exist();
						parser.store_peak_token(_gi.byoyomi);
					}
				}

				if (!_gi.check_state()) {
					std::cout << "Error: invalid go cmd = [" << cmd << "]" << std::endl;
					throw std::runtime_error("Error");
				}

				// 持ち時間制御
				// https://zenn.dev/reputeless/books/standard-cpp-for-competitive-programming/viewer/library-algorithm
				const auto&& turn = _board.turn();
				const int&& n_self_ply_total = _board.ply() / 2;    // 今までに自分が指した手の数
				//const int& div = std::max(
				//	{
				//	    div_init - (n_self_ply_total - 1) * 2,
				//	    div_second - (n_self_ply_total - n_self_ply_while_begining - 1),
				//    	div_final
				//	}
				//);

				const int& div = std::max(
					{
						div_init - (n_self_ply_total - 1),
						div_final
					}
				);

				const int&& base_time = (_gi.get_time(turn) / div);
				if (_gi.is_inc()) {
					_time_limit = base_time + _gi.get_inc(turn) - _network_delay;
				}
				else if (_gi.is_byoyomi()) {
					_time_limit = base_time + _gi.byoyomi - _network_delay;
				}
				else if (_gi.is_extra_time_zero()) {
					// HACK: 一応0でclip してあるけど、こうなったら正直投了。ただ、切れ負け自体やること無いと思うので適当に。
					_time_limit = my_max(base_time - _network_delay, 0);
				}
				else if (_gi.is_infinite) {
					_time_limit = 0;
				}
				else {
					std::cout << "Error: got unexpected instruction about time." << std::endl;;
					throw std::runtime_error("Error");
				}
				_n_extend_time = 0;
				std::cout << "info string div = " << div << std::endl;
				std::cout << "info string time_limit = " << _time_limit << std::endl;

				// TODO: ここ、見た目的になんか書いておきたいけど...
				_stop_searcher_thread_if_running();
				_set_uct_search_start();

				// https://tadaoyamaoka.hatenablog.com/entry/2018/02/20/223237
				_searcher_thread = std::thread([this]() {
					this->go();
				});
			}
			else {
				std::cout << "ParserError: parser got unexpected cmd == [" << cmd << "]" << std::endl;;
				//throw std::runtime_error("Error");
			}
		}
	}

	void usi() {
		
		std::cout << "id name PuctPlayerV0.1.0" << std::endl;
		std::cout << "id author lefu777" << std::endl;
		std::cout << "option name " << "USI_Ponder" << " type " << "check" << " default " << (false ? "true" : "false") << std::endl;
		std::cout << "option name " << "threads" << " type " << "spin" << " default " << _n_threads << " min " << 1 << " max " << 16 << std::endl;
		std::cout << "option name " << "batch_size" << " type " << "spin" << " default " << _batch_size << " min " << 0 << " max " << INT_MAX << std::endl;
		std::cout << "option name " << "minimum_thinking_time" << " type " << "spin" << " default " << _minimum_thinking_time << " min " << 0 << " max " << INT_MAX << std::endl;
		std::cout << "option name " << "network_delay" << " type " << "spin" << " default " << _network_delay << " min " << 0 << " max " << INT_MAX << std::endl;
		std::cout << "option name " << "const_playout" << " type " << "spin" << " default " << _const_playout << " min " << 0 << " max " << INT_MAX << std::endl;
		std::cout << "option name " << "print_pv_per_playout" << " type " << "spin" << " default " << _print_pv_per_playout << " min " << 0 << " max " << INT_MAX << std::endl;
		std::cout << "option name " << "book_moves" << " type " << "spin" << " default " << _book_moves << " min " << 0 << " max " << INT_MAX << std::endl;
		std::cout << "option name " << "onnx_model" << " type " << "string" << " default " << _onnx_model << std::endl;
		std::cout << "option name " << "book_file" << " type " << "string" << " default " << _book_filename << std::endl;
		std::cout << "usiok" << std::endl;
	}

	// @arg name
	//     : setoption name <id> value <x>
	//       setoption は、usi_parser で取り除かれている
	void setoption(std::string cmd) {
		const auto&& tokens = tokenize(cmd);
		if (tokens[1] == "threads") {
			_n_threads = std::atoi(tokens[3].c_str());
		}
		else if (tokens[1] == "batch_size") {
			_batch_size = std::atoi(tokens[3].c_str());
		}
		
		else if (tokens[1] == "minimum_thinking_time") {
			_minimum_thinking_time = std::atoi(tokens[3].c_str());
		}
		else if (tokens[1] == "network_delay") {
			_network_delay = std::atoi(tokens[3].c_str());
		}
		else if (tokens[1] == "const_playout") {
			_const_playout = std::atoi(tokens[3].c_str());
		}
		else if (tokens[1] == "onnx_model") {
			_onnx_model = tokens[3];
		}
		else if (tokens[1] == "book_file") {
			_book_filename = tokens[3];
		}
		else if (tokens[1] == "print_pv_per_playout") {
			_print_pv_per_playout = std::atoi(tokens[3].c_str());
		}
		else if (tokens[1] == "book_moves") {
			_book_moves = std::atoi(tokens[3].c_str());
		}
		else {
			std::cout << "Warning: unexpected setoption name = " << tokens[1] << std::endl;
		}
	}

	// 探索を終了すべきか否か
	// @return
	//     : true : 探索を終了する
	//       false: 探索を続行
	bool should_exit_search() {
		if (_const_playout > 0) {    // プレイアウト数制限が有効である
			return (_playout_count >= _const_playout);
		}

		const auto& elapsed_time_ms = _stop_watch.elapsed_ms_interim();

		if (elapsed_time_ms < _minimum_thinking_time) {
			return false;
		}

		// NOTE
		//     : 探索木を再利用しているので、mvoe_count0 - mvoe_count1 が、
		//       見かけの思考時間に見合わない(思考時間に対して大きすぎる)場合がある。
		//       そういう時は、time_limit の半分も使わずに 探索が切り上げられる事がある。
		// 残りの探索で、次善手が最善手を超える可能性があるなら、探索は続行
		const int&& estimated_extra_playout = _playout_count * (_time_limit - elapsed_time_ms) / elapsed_time_ms;

		const auto& crrt_head = global_tree->current_head;
		const auto&& idxs = crrt_head->get_sorted_idx_list();
		const auto&& mvoe_count0 = crrt_head->get_move_count(idxs[0]);
		const auto&& mvoe_count1 = crrt_head->get_move_count(idxs[1]);
		const auto&& winrate0 = crrt_head->get_winrate(idxs[0]);
		const auto&& winrate1 = crrt_head->get_winrate(idxs[1]);
		const auto&& move0 = crrt_head->get_move(idxs[0]);
		const auto&& move1 = crrt_head->get_move(idxs[1]);
		if (mvoe_count0 - mvoe_count1 <= estimated_extra_playout) {
			return false;
		}

		// HACK: 1.5, 0.1, 20 がマジックナンバーになってるのでconstexprして！
		// まだ次善手が逆転可能と思われる割合
		double comebackable_rate = 1.5;
		if (winrate0 < winrate1) {
			// 勝率が勝っているなら、逆転はよりしやすいはず。
			comebackable_rate += 0.1;
		}

		// 逆転出来なさそうになっても、一度はチャンスがある。
		// 探索延長したことが無い &&
		// 21手目以降 && 
		// 2倍に探索延長するだけの時間の余裕がある &&
		// まだ逆転出来る範疇である
		if (
			_n_extend_time == 0 && 
			_board.ply() > 20 &&
			_time_limit * 2 < _gi.get_time(_board.turn()) &&
			mvoe_count0 < mvoe_count1 * comebackable_rate
			) {
			_time_limit *= 2;
			_n_extend_time += 1;
			std::cout << "info string extend time, "
				<< "[" << __move_to_usi(move0.value()) << ", " << __move_to_usi(move1.value()) << "]"
				<< "[" << mvoe_count0 << ", " << mvoe_count1 << "]"
    			<< "[" << winrate0 << ", " << winrate1 << "]" << std::endl;
			//crrt_head->print_child_move_counts();
			return false;
		}

		std::cout << "info string pruning, "
			<< "[" << __move_to_usi(move0.value()) << ", " << __move_to_usi(move1.value()) << "]"
			<< "[" << mvoe_count0 << ", " << mvoe_count1 << "]"
			<< "[" << winrate0 << ", " << winrate1 << "]"
			<< "[" << estimated_extra_playout << ", " << _playout_count << "]"
			<< std::endl;

		// 二度目のチャンスはない。
		return true;
	}

	// TODO: 本当は、ここでinit とか呼ぶ。
	bool _isready() {
		_nn = new NNTensorRT(_onnx_model, _batch_size);
		if (_nn == nullptr) {
			return false;
		}
		if (_input != nullptr) {    // これ、Free でcheck してくれると思うけど、まぁ一応。
			cudaFree(_input);
			cudaFree(_output_policy);
			cudaFree(_output_value);
		}
		cudaHostAlloc((void**)&_input, sizeof(float) * N_FEATURE_WHC * _batch_size, cudaHostAllocPortable);
		cudaHostAlloc((void**)&_output_policy, sizeof(float) * N_LABEL_SIZE * _batch_size, cudaHostAllocPortable);
		cudaHostAlloc((void**)&_output_value, sizeof(float) * 1 * _batch_size, cudaHostAllocPortable);

		_eval_queue = std::vector<UctNodeAndFeatureAndColor>(_batch_size);
		_current_queue_idx = 0;
		for (int i = 0; i < _batch_size; ++i) {
			auto&& e = _eval_queue[i];
			e.first = nullptr;
			e.second.first = _input + N_FEATURE_WHC * i;    // i番目のbatch に対応したメモリ領域を紐づける。
		}

		_board.reset();

		if (!yobook_parser.parse(_book_filename)) {
			return false;
		}

#ifdef DEBUG
		std::cout << "info string yobook_parser.size() = [" << yobook_parser.size() << "]" << std::endl;
#endif

		return true;
	}

	void isready() {
		if (_isready()) {
			std::cout << "readyok" << std::endl;
		}
		else {
			std::cout << "Error: isready failed" << std::endl;
			throw std::runtime_error("Error");
		}
	}

	void _usinewgame() {
		;
	}

	void usinewgame() {
		_usinewgame();
		//std::cout << "info got cmd = usinewgame" << std::endl;
	}

	// @arg moves
	//     : 以下の2通りがある。
	//       startpos moves xxxx xxxx ...
	//       [sfen形式の任意の初期局面(先頭にsfenは無い)] moves xxxx xxxx ...
	virtual void position(const std::string& position_str) {
		//std::cout << "info got cmd = position" << std::endl;
		//if (!_board.set_position(moves)) {
		//	std::cout << "info Error: failed to set_position" << std::endl;
		//}
		_position_str = position_str;
	}

	void go() {
		_stop_watch.start();

		std::string startsfen;
		std::vector<Move> moves;
		Key startpos_key;
		_board.set_position_and_get_components(_position_str, &startsfen, &moves, &startpos_key);
		bool found = global_tree->reset_to_position(startpos_key, moves);
		std::cout << "info string found_old_head = " << (found ? "true" : "false") << std::endl;



		const auto&& mate_move3 = _board.mateMove(3);
		//const auto&& mate_move3 = 0;
		//const auto&& mate_move1 = _board.mateMoveIn1Ply();

#ifdef DEBUG_
		std::cout << "[" << _board.dump() << "]" << std::endl;
		std::cout << "[" << mate_move1 << "]m1" << std::endl;
		std::cout << "[" << __move_to_usi(mate_move1) << "]m1" << std::endl;
		std::cout << "[" << mate_move3 << "]m3" << std::endl;
		std::cout << "[" << __move_to_usi(mate_move3) << "]m3" << std::endl;
#endif

		if (_board.is_game_over()) {
			std::cout << "bestmove resign" << std::endl;
		}
		else if (_board.is_nyugyoku()) {
			std::cout << "bestmove win" << std::endl;
		}
		else if (mate_move3) {
			const auto bestmove = __move_to_usi(mate_move3);
			std::cout << "info score mate + pv " << bestmove << std::endl;
			std::cout << "bestmove " << bestmove << std::endl;
		}
		else {
			const auto bestmove = __move_to_usi(_go());
			std::cout << "bestmove " << bestmove << std::endl;

#ifdef DEBUG
			const auto& root_node = global_tree->current_head;
			for (int i = 0; i < root_node->legal_moves_size; ++i) {
				std::cout << i << ":" << __move_to_usi(root_node->child_moves[i].value())
					<< " move_count:" << root_node->child_move_counts[i]
					<< " nn_rate:" << root_node->child_probabilitys[i]
					<< "win_rate:"
					<< (root_node->child_move_counts[i] == 0 ? 0 :
						root_node->child_sum_values[i] / root_node->child_move_counts[i])
					<< std::endl;
			}

			print_ucb(root_node);
#endif

		}

		_stop_watch.stop();
	}

	// https://github.com/TadaoYamaoka/python-dlshogi/blob/master/pydlshogi/player/mcts_player.py
	// 終局 or 宣言勝ち の局面では呼び出さない。
	int _go() {
#ifdef DEBUG_
		std::cout << "info string debug: _go() start" << std::endl;
#endif
		// _check_state_info(_board, "_go()");

		assert(!(_board.is_game_over() || _board.is_nyugyoku()));

		// プレイアウト数をクリア
		_playout_count = 0;
		_last_pv_print_playout = 0;

		// ルートノードが未展開なら展開する
		if (global_tree->current_head->legal_moves_size < 0) {    // 未展開
			global_tree->current_head->expand_node(_board);
		}

		// 候補手が一つの場合は、その手を返す。
		__LegalMoveList legal_moves(_board);
		if (legal_moves.size() == 1) {
			return legal_moves.move();
		}

		//std::cout << "info string ply = " << _board.ply() << std::endl;
		if (_board.ply() <= _book_moves) {
			//std::cout << "info string _board.toSFEN() = [" << _board.toSFEN() << "]" << std::endl;
			const auto& find_status = yobook_parser.find(_board.toSFEN());
			if (find_status.first == true) {    // 定跡にhit
				const yo_book_value_t& book_hit = (*(find_status.second)).second;
				const yo_book_move_t& bestmove = book_hit.get_best_by_eval();
				bestmove.print_usi();
				return bestmove.move1;
			}
		}

		// TODO: これ微妙では？仮に詰んでるなら、先に詰みcheck 必要じゃないかな？
        // 評価されていないなら評価する
		if (!global_tree->current_head->is_evaled()) {
			queue_node(global_tree->current_head, _board);
			eval_node();
		}
		// 探索
		search();

		// 結果を取得して表示
		int bestmove;
		float bestvalue;
		get_bestmove_and_print_pv(&bestmove, &bestvalue);

		// TODO: ここでbestvalue の値によっては投了

		return bestmove;
	}



	// UCTSearcher::ParallelUctSearch()
	void search() {
		// root が未評価なら先に評価する。
#ifdef DEBUG_
		std::cout << "DEBUG: global_tree->current_head->is_evaled() = "
			<< global_tree->current_head->is_evaled() << std::endl;
		std::cout << "DEBUG: global_tree->current_head->legal_moves_size = "
			<< global_tree->current_head->legal_moves_size << std::endl;
		std::cout << "DEBUG: global_tree->current_head " << global_tree->current_head << std::endl;
#endif
#ifdef DEBUG_
		std::cout << "DEBUG: value = " << global_tree->current_head->value << std::endl;
#endif

		// NOTE
		//     : uct_search() で、一旦全部trajectories_batch に入る。
		//     : - QUEUING はtrajectories_batch へ。推論後にbackup する為。
		//       - DISCARDED はtrajectories_batch_discarded へ。VIRTUAL_LOSS を元に戻すため。
		//       - それ以外(勝敗が明確に分かった場合) はどこにも残らない。
		std::vector<Trajectory> trajectories_batch;
		std::vector<Trajectory> trajectories_batch_discarded;
		int current_batch_top;

		while (1) {
			//std::cout << "[" << _playout_count << "]" << std::endl;
			trajectories_batch.clear();
			trajectories_batch_discarded.clear();
			current_batch_top = -1;    // top の要素のidx
			//std::vector<__Board> board_cpys(_batch_size, _board);
			for (int i = 0; i < _batch_size; ++i) {
				// TODO: これ、本当はboard は参照渡しして、uct_search() でundoMove() (というより、pop())もするようにすべき。
				// 明示的なコピーコンストラクタの呼出し
#ifdef DEBUG_
				std::cout << "info: start copy constructer" << std::endl;
#endif
				//_check_state_info(_board, "search() (before cpy)");	
				// NOTE: やっぱこれ、local だったのが良くなかったんやで。
				__Board board_cpy(_board);
				//__Board& board_cpy = board_cpys[i];
				// _check_state_info(board_cpy, "search() (after cpy)");
#ifdef DEBUG_
				std::cout << "info: end copy constructer" << std::endl;
#endif
				// TODO: これで本当にからのtrajectory 入ってるかは怪しい。
				trajectories_batch.emplace_back(std::initializer_list<UctNodeAndMove>{});
				++current_batch_top;

				float result = uct_search(board_cpy, global_tree->current_head, trajectories_batch[current_batch_top]);

#ifdef DEBUG_
				std::cout << "info debug: for[" << i << "] done, result = " << result << std::endl;
#endif

				if (result == RESULT_DISCARDED) {
					trajectories_batch_discarded.emplace_back(trajectories_batch[current_batch_top]);
				}
				else {
					++_playout_count;
				}
				if (result != RESULT_QUEUING) {    // 推論待ち出なければ、既にbackup済みなので削除。
					trajectories_batch.pop_back();
					--current_batch_top;
				}
				if (trajectories_batch_discarded.size() > _batch_size / 2) {    // batch_size の半分以上を破棄が占める場合、とっとと評価フェーズに移行。
					break;
				}
			}
#ifdef DEBUG_
			std::cout << "_playout_count = " << _playout_count << std::endl;
			//std::cout << "info debug: for all done" << std::endl;
#endif
#ifdef DEBUG_
			std::cout << "info string debug: eval_node() start" << std::endl;
#endif

			// 評価
			if (trajectories_batch.size() > 0) {
				eval_node();
			}

#ifdef DEBUG_
			std::cout << "info string debug: eval_node() done" << std::endl;
#endif

			// Virtual Loss を元に戻す 
			for (const auto& trajectory : trajectories_batch_discarded) {
				for (const auto& nm : trajectory) {
					nm.first->move_count -= VIRTUAL_LOSS;
					nm.first->child_move_counts[nm.second] -= VIRTUAL_LOSS;
				}
			}

			// NOTE: queueing だった各trajectory について、探索結果を反映していく。
			// backup
			float result;
			for (const auto& trj : trajectories_batch) {
				result = -1;    // TOOD: RESULT_NONE = -1 として他の値と衝突しないか。
				//       (まぁ、この初期値は一切使ってないから、そもそも初期化せんでええ気もする。)
				for (auto it = trj.rbegin(), it_rend = trj.rend(); it != it_rend; ++it) {
					auto node = it->first;
					auto next_index = it->second;
					if (it == trj.rbegin()) {
						// このvalue は、今回の推論によるものである。
						// queue_node(child_node, board); としてることからも分かるように、
						// 今回の推論によるものは子ノードの手番からのvalue なので反転する。
						result = 1 - node->child_nodes[next_index]->value;
					}
					update_result(node, next_index, result);
					result = 1 - result;    // NOTE: 手番に応じて反転？(@20230819)
				}
			}

			if (_is_stop) {
				break;
			}

			// 探索を打ち切るか確認
			if (should_exit_search()) {
				break;
			}

			// pv表示
			if (_print_pv_per_playout > 0) {
				if (_playout_count - _last_pv_print_playout >= _print_pv_per_playout) {
					_last_pv_print_playout = _playout_count;
					get_bestmove_and_print_pv(nullptr, nullptr);
				}
			}

		}
	}

	// NOTE
	//     : result は、current_node の手番からみた結果である。
	float uct_search(__Board& board, UctNode* current_node, Trajectory& trajectory) {
#ifdef DEBUG_
		std::cout << "info debug: uct_search() start" << std::endl;
#endif
		if (current_node->legal_moves_size < 0) {    // 未展開
#ifdef DEBUG_
			std::cout << "DEBUG: [unexpanded] current_node = " << current_node << std::endl;
#endif
			current_node->expand_node(board);
		}
		else {
#ifdef DEBUG_
			std::cout << "DEBUG: [already expanded] current_node = " << current_node << std::endl;
#endif
		}

#ifdef DEBUG_
		std::cout << "DEBUG: " << board.toSFEN() << std::endl;
#endif
		int next_index = select_max_ucb_child(current_node);
		board.push(current_node->child_moves[next_index].value());

#ifdef DEBUG_
		std::cout << "[" << _playout_count << "] "
			<< __move_to_usi(current_node->child_moves[next_index].value()) << std::endl;
#endif

		// Virtual Loss を加算
		current_node->move_count += VIRTUAL_LOSS;
		current_node->child_move_counts[next_index] += VIRTUAL_LOSS;

		// 探索経路を記録
		trajectory.emplace_back(current_node, next_index);

		// NOTE
		//     : 子ノードの実体がない 
		//       <=> current_node->child_nodes[next_index]->value は無い
		//       <=> 子ノードは未評価である
		//    -> 勝敗が明確でない限り、評価する。
		//     : 以下のresult は、当然だが、current_node の手番側から見た結果である。
		// TODO
		//     : 本当はVALUE_NONE とかで初期化すべきだけど、uct_node の奴を使うのなんかきもくねぇか。
		//    -> どっかに纏めたい。constant.hpp/.cpp として。
		float result = VALUE_NONE;
		if (current_node->child_nodes[next_index] == nullptr) {    // child_nodes[next_index]の実体がない -> 実体を作成して評価
#ifdef DEBUG_
			std::cout << "info debug: uct_search() child_node[next_idx] == nullptr" << std::endl;
#endif
			UctNode* child_node = current_node->create_child_node(next_index);

#ifdef DEBUG_
			auto print_draw_state = [&board](const RepetitionType rt) {
				    switch (rt) {
				    case RepetitionDraw:
						std::cout << "RepetitionDraw[" << board.toSFEN() << "]" << std::endl;
		    			break;
			    	case RepetitionWin: case RepetitionSuperior:
						std::cout << "RepetitionWin or RepetitionSuperior[" << board.toSFEN() << "]" << std::endl;
	    				break;
		    		case RepetitionLose: case RepetitionInferior:
						std::cout << "RepetitionLose or RepetitionInferior[" << board.toSFEN() << "]" << std::endl;
					    break;
					case NotRepetition:
						//std::cout << "NotRepetition" << std::endl;
						break;
    				default:
						std::cout << "Error: print_draw_state() got default" << std::endl;
						throw std::runtime_error(" ");
						break;
	    			}
				};
#endif

			// child_node の局面において引き分けをチェック。
			auto repetition_type = board.isDraw(INT_MAX);

#ifdef DEBUG_
			print_draw_state(repetition_type);
#endif

			if (repetition_type == NotRepetition) {
				if (board.is_nyugyoku() || board.mateMove(3)) {
					child_node->value = VALUE_WIN;
					result = 0;
					//std::cout << "[" << _playout_count << "] result = Nyugyoku or Mate" << std::endl;
				}
				else {
					child_node->expand_node(board);
					// 以下は同じ意味。
#if 1
					if (child_node->legal_moves_size == 0) {
#else
					if (board.is_game_over()) {
#endif
						child_node->value = VALUE_LOSE;
						result = 1;
					}
					else {
#ifdef DEBUG_
						std::cout << "[" << _playout_count << "] queue_node()" << std::endl;
#endif
						// TODO: queue に追加。(board はコピーするのか、は割と疑問。その場で必要な情報だけ抜き取る感じかな～)
						// _check_state_info(board, "uct_search() (before queue_node())");
						queue_node(child_node, board);
						// _check_state_info(board, "uct_search() (after queue_node())");
						//std::cout << "info debug: uct_search() return RESULT_QUEUING;" << std::endl;
						return RESULT_QUEUING;
					}
					}
				}
			else {    // 引き分け
				//std::cout << "[" << _playout_count << "] result = Repetition" << std::endl;
				switch (repetition_type) {
				case RepetitionDraw:
					child_node->value = VALUE_DRAW;
					result = 0.5;
#ifdef DEBUG
					//std::cout << "[" << _playout_count << "] set result = " << result << std::endl;
#endif
					break;
				case RepetitionWin: case RepetitionSuperior:
					child_node->value = VALUE_WIN;
					result = 0;
					break;
				case RepetitionLose: case RepetitionInferior:
					child_node->value = VALUE_LOSE;
					result = 1;
					break;
				default:
#ifdef DEBUG
					std::cout << "Error: got unexpected repetition_type = " << repetition_type << std::endl;
#endif
					throw std::runtime_error("Error");
				}
			}
			// queue_node() 以外であれば、勝敗が決しているので評価済みに。
			child_node->set_evaled();
		}
		else {    // child_nodes[next_index]の実体がある。 -> 評価されている

#ifdef DEBUG_
			std::cout << "info debug: uct_search() child_node[next_idx] != nullptr" << std::endl;
#endif
			UctNode* next_node = current_node->child_nodes[next_index].get();
			assert(next_node != nullptr);
			if (next_node->value == VALUE_NONE) {    // 実体があるのに評価が無い -> 評価待ち(RESULT_QUEUING)
				//std::cout << "info debug: uct_search() return RESULT_DISCARDED;" << std::endl;
				return RESULT_DISCARDED;
			}

			if (next_node->value == VALUE_WIN) {

#ifdef DEBUG_
				std::cout << "info debug: uct_search() next_node->value = VALUE_WIN" << std::endl;
#endif
				result = 0;
			}
			else if (next_node->value == VALUE_LOSE) {
				result = 1;
			}
			else if (next_node->value == VALUE_DRAW) {
				result = 0.5;
			}
			// HACK: これは要らないはず。VALUE_LOSE と同義のはず。でも、python-dlshogi2 にあったので。
#ifdef DEBUG
			else if (next_node->legal_moves_size == 0) {
				std::cout << "info string Warning: at uct_search(), next_node->value != VALUE_LOSE, but next_node->legal_moves_size == 0" << std::endl;
				result = 1;
			}
#endif
			else {
				result = this->uct_search(board, next_node, trajectory);
				if (result == RESULT_QUEUING || result == RESULT_DISCARDED) {
					return result;
				}
			}
		}
		// HACK: これは要らないはず。でも、python-dlshogi2 にあったので。
		if (result == RESULT_QUEUING || result == RESULT_DISCARDED) {
			std::cout << "info string Warning: at uct_search(), although not after recursion, the result is RESULT_QUEUING or RESULT_DISCARDED" << std::endl;
			return result;
		}

		// NOTE
		//     : queueing ではない場合、backup(探索木への探索結果の反映) はここで行う。
		//       (backup は、queueing の場合はeval_node()後に行い、
		//       discarded の場合はそもそもbackup は無理だしする必要もなく、それ以外の勝敗が決している場合はここで行う。)
		update_result(current_node, next_index, result);

		return 1 - result;
	}

	// 一番探索すべき指し手のindex を返す。
	int select_max_ucb_child(UctNode * node) const {
#ifndef PYTHON_DLSHOGI2
		const int& total = node->legal_moves_size;

		std::vector<int> current_max_idx;
		float reward_mean, bonus;
		float current_max_ucb_score = -1;    // スコアが負にならない前提。

		const float sqrt_sum = sqrt(node->move_count);
		const float C_s = logf((1.0 + node->move_count + puct_c_base) / puct_c_base) + puct_c_init;

		const float init_bonus = (node->move_count == 0 ? 1.0 : sqrt_sum);

		for (int i = 0; i < total; ++i) {
			if (node->child_move_counts[i] == 0) {
				// TODO: 親ノードの平均報酬で近似する。
				reward_mean = 0;
				bonus = init_bonus;
			}
			else {
				reward_mean = node->child_sum_values[i] / node->child_move_counts[i];
				bonus = sqrt_sum / (1 + node->child_move_counts[i]);
			}
			float ucb_score = reward_mean + C_s *  node->child_probabilitys[i] * bonus;

#ifdef DEBUG_
			std::cout << "[" << i << "] :";
			std::cout << " " << reward_mean << ", " << bonus;
			std::cout << ", " << ucb_score << ", " << current_max_ucb_score << std::endl;
#endif

			if (ucb_score > current_max_ucb_score) {
				current_max_ucb_score = ucb_score;
				current_max_idx.clear();
				current_max_idx.emplace_back(i);
			}
			else if (ucb_score == current_max_ucb_score) {
				current_max_idx.emplace_back(i);
			}
#ifdef DEBUG_
			std::cout << reward_mean << ", " << bonus << std::endl;;
			std::cout << current_max_idx << ", " << current_max << std::endl;;
#endif
		}

#ifdef DEBUG_
		std::cout << "final = " << current_max_idx << ", " << current_max << std::endl;;
#endif

		// この、初動の全くデータがなくてrandom に選ぶ時の奴、
		// 指し手のオーダリングとかしたらちょっと良くなりそう。
		// (スコアが同じ奴らの、オーダリングにおけるスコアを計算して、一番高い奴を選ぶ。)
		return current_max_idx[mt_genrand_int32() % current_max_idx.size()];
#else
		const int& total = node->legal_moves_size;
#ifdef DEBUG_
		assert(total > 0 && "Warning: this node may be unexpanded.");
		std::cout << (node == nullptr ? "nullptr" : "not null") << std::endl;
		std::cout << node->legal_moves_size << std::endl;
#endif
#ifdef DEBUG_
		std::cout << "DEBUG: node = " << node << std::endl;
#endif
		std::vector<int> current_max_idx;
		float reward_mean, bonus;
		float current_max_ucb_score = -1;    // スコアが負にならない前提。

		const float sqrt_sum = sqrt(node->move_count);

		for (int i = 0; i < total; ++i) {
			if (node->child_move_counts[i] == 0) {
				reward_mean = 0;
			}
			else {
				reward_mean = node->child_sum_values[i] / node->child_move_counts[i];
			}
			if (node->move_count == 0) {
				bonus = 1;
			}
			else {
				bonus = sqrt_sum / (1 + node->child_move_counts[i]);
			}
			float ucb_score = reward_mean + puct_c_init * node->child_probabilitys[i] * bonus;

#ifdef DEBUG_
			std::cout << "[" << i << "] :";
			std::cout << " " << reward_mean << ", " << bonus;
			std::cout << ", " << ucb_score << ", " << current_max_ucb_score << std::endl;
#endif

			if (ucb_score > current_max_ucb_score) {
				current_max_ucb_score = ucb_score;
				current_max_idx.clear();
				current_max_idx.emplace_back(i);
			}
			else if (ucb_score == current_max_ucb_score) {
				current_max_idx.emplace_back(i);
			}
#ifdef DEBUG_
			std::cout << reward_mean << ", " << bonus << std::endl;;
			std::cout << current_max_idx << ", " << current_max << std::endl;;
#endif
		}

#ifdef DEBUG_
		std::cout << "final = " << current_max_idx << ", " << current_max << std::endl;;
#endif

		// この、初動の全くデータがなくてrandom に選ぶ時の奴、
		// 指し手のオーダリングとかしたらちょっと良くなりそう。
		// (スコアが同じ奴らの、オーダリングにおけるスコアを計算して、一番高い奴を選ぶ。)
		return current_max_idx[mt_genrand_int32() % current_max_idx.size()];
#endif
	}

	// 一番探索すべき指し手のindex を返す。
	void print_ucb(UctNode* node) const {
#ifndef PYTHON_DLSHOGI2
		const int& total = node->legal_moves_size;
		std::vector<float> ucbvec;

		std::vector<int> current_max_idx;
		float reward_mean, bonus;
		float current_max_ucb_score = -1;    // スコアが負にならない前提。

		const float sqrt_sum = sqrt(node->move_count);
		const float C_s = logf((1.0 + node->move_count + puct_c_base) / puct_c_base) + puct_c_init;

		const float init_bonus = (node->move_count == 0 ? 1.0 : sqrt_sum);

		for (int i = 0; i < total; ++i) {
			if (node->child_move_counts[i] == 0) {
				// TODO: 親ノードの平均報酬で近似する。
				reward_mean = 0.5;
			}
			else {
				reward_mean = node->child_sum_values[i] / node->child_move_counts[i];
			}
			if (node->move_count == 0) {
				// log0 の回避
				bonus = init_bonus;
			}
			else {
				bonus = sqrt_sum / (1 + node->child_move_counts[i]);
			}
			float ucb_score = reward_mean + C_s * node->child_probabilitys[i] * bonus;

			ucbvec.emplace_back(ucb_score);
#ifdef DEBUG_
			std::cout << "[" << i << "] :";
			std::cout << " " << reward_mean << ", " << bonus;
			std::cout << ", " << ucb_score << ", " << current_max_ucb_score << std::endl;
#endif

			if (ucb_score > current_max_ucb_score) {
				current_max_ucb_score = ucb_score;
				current_max_idx.clear();
				current_max_idx.emplace_back(i);
			}
			else if (ucb_score == current_max_ucb_score) {
				current_max_idx.emplace_back(i);
			}
		}
#else
		std::vector<float> ucbvec;
		const int& total = node->legal_moves_size;
#ifdef DEBUG_
		assert(total > 0 && "Warning: this node may be unexpanded.");
		std::cout << (node == nullptr ? "nullptr" : "not null") << std::endl;
		std::cout << node->legal_moves_size << std::endl;
#endif
#ifdef DEBUG_
		std::cout << "DEBUG: node = " << node << std::endl;
#endif
		std::vector<int> current_max_idx;
		float reward_mean, bonus;
		float current_max_ucb_score = -1;    // スコアが負にならない前提。

		const float sqrt_sum = sqrt(node->move_count);

		for (int i = 0; i < total; ++i) {
			if (node->child_move_counts[i] == 0) {
				reward_mean = 0;
			}
			else {
				reward_mean = node->child_sum_values[i] / node->child_move_counts[i];
			}
			if (node->move_count == 0) {
				bonus = 1;
			}
			else {
				bonus = sqrt_sum / (1 + node->child_move_counts[i]);
			}
			float ucb_score = reward_mean + puct_c_init * node->child_probabilitys[i] * bonus;
			ucbvec.emplace_back(ucb_score);

#ifdef DEBUG
			std::cout << "[" << i << "] :";
			std::cout << " " << reward_mean << ", " << bonus;
			std::cout << ", " << ucb_score << ", " << current_max_ucb_score << std::endl;
#endif

			if (ucb_score > current_max_ucb_score) {
				current_max_ucb_score = ucb_score;
				current_max_idx.clear();
				current_max_idx.emplace_back(i);
			}
			else if (ucb_score == current_max_ucb_score) {
				current_max_idx.emplace_back(i);
			}
#ifdef DEBUG_
			std::cout << reward_mean << ", " << bonus << std::endl;;
			std::cout << current_max_idx << ", " << current_max << std::endl;;
#endif
		}
#endif
		for (int i = 0; i < total; ++i) {
			std::cout << "[" << i << ":" << __move_to_usi(node->child_moves[i].value()) << "] " << ucbvec[i] << std::endl;
		}
	}

	void update_result(UctNode * node, int next_index, float result) {
		node->sum_value += result;
		node->move_count += 1 - VIRTUAL_LOSS;
		node->child_sum_values[next_index] += result;
		node->child_move_counts[next_index] += 1 - VIRTUAL_LOSS;
	}

	// TODO
	//     : とりま参照無くしてみたけど意味ないやろうな。stateinfo 壊れる問題どうにかなおさんと。。。
	//     : StartState が壊れてるなら、やっぱりStartState が実体としてそのままclass メンバーになってるので、
	//       クラスが消えちゃうと、そのメンバーも消えちゃう。
	//       for文抜けたらオブジェクトも解放されるはずで、もし仮にqueueに参照が入ってれば、
	//       そのqueueに入ってるboard のStartState は無くなってる。
	//       でも、queue に参照が入ってるようにはみえない(コピーコンストラクタがenqueue の瞬間呼ばれてるか否かを確認。)
	//       (アドレスでも良いけど、ちょっと確認面倒。)
	// @arg node: 未評価のnode
	// @arg board: node に対応した局面
	void queue_node(UctNode * node, __Board & board) {
		// _check_state_info(board, "queue_node() (ref arg)");
		auto& queue_top = _eval_queue[_current_queue_idx];
		queue_top.first = node;
		board.make_input_features(queue_top.second.first);
		queue_top.second.second = board.turn();

		++_current_queue_idx;
	}

	// queue の中身を一斉に評価
	void eval_node() {
#ifdef DEBUG_
		std::cout << "info string debug: _nn->forward() start" << std::endl;
		std::cout << "info string debug: _current_queue_idx = " << _current_queue_idx << std::endl;
#endif
		_nn->forward(_current_queue_idx, _input, _output_policy, _output_value);
#ifdef DEBUG_
		std::cout << "info string debug: _nn->forward() done" << std::endl;
#endif
		// TODO
		//     : policy を追加
		//       全ての合法手を走査して、move_to_label() 使えばok
		for (int i = 0; i < _current_queue_idx; ++i) {
			auto&& e = _eval_queue[i];
			auto node = e.first;
			auto&& color = e.second.second;

#ifdef DEBUG_
			std::cout << "info string debug: _output_value[" << i << "] = " << _output_value[i] << std::endl;
			std::cout << "DEBUG: node = " << node << std::endl;
#endif
			node->value = sigmoid(_output_value[i]);

#ifdef DEBUG_
			std::cout << "info string debug: node->value = " << node->value << std::endl;
#endif

			for (int j = 0; j < node->legal_moves_size; ++j) {
				const int&& label = move_to_label(node->child_moves[j], color);
				// i番目のbatch から、ラベル番号 = labelの推論結果を取り出す
				const auto& logit = _output_policy[N_LABEL_SIZE * i + label];
//#ifdef DEBUG
//				std::cout << "[" << j << ":" << __move_to_usi(node->child_moves[j].value()) << "] label = "
//					<< label << ", logit = " << logit << std::endl;
//				if (label >= N_LABEL_SIZE) {
//					std::cout << "Error: illegal label = " << label << std::endl;
//					throw std::runtime_error("Error");
//				}
//#endif
				node->child_probabilitys[j] = logit;
			}
			// softmax
			softmax_with_temperature(node, 1);

#ifdef DEBUG_
			for (int j = 0; j < node->legal_moves_size; ++j) {
				const int&& label = move_to_label(node->child_moves[j], color);
				// i番目のbatch から、ラベル番号 = labelの推論結果を取り出す
				const auto& logit = _output_policy[N_LABEL_SIZE * i + label];
				std::cout << "[" << j << ":" << __move_to_usi(node->child_moves[j].value()) << "]" << 
					"label = " << label
					<< ", logit = " << logit
					<< ", policy = "<< node->child_probabilitys[j]
					<< std::endl;
				if (label >= N_LABEL_SIZE) {
					std::cout << "Error: illegal label = " << label << std::endl;
					throw std::runtime_error("Error");
				}
			}
#endif

			node->set_evaled();

#ifdef DEBUG_
			std::cout << "updated value = " << _eval_queue[i].first->value << std::endl;
#endif
		}

		_eval_queue.clear();
		_current_queue_idx = 0;
	}

	// @arg ptr_bestmove: bestmove が欲しいときは、ここにbestmove を貰うポインタを置く。
	// @arg ptr_bestvalue: bestvalue が欲しいときは、ここにbestvalue を貰うポインタを置く。
	void get_bestmove_and_print_pv(int* ptr_bestmove, float* ptr_bestvalue) {
		// 訪問回数が最大の手を選択する。
		int bestmove_idx = global_tree->current_head->get_bestmove_idx();
		float bestvalue = global_tree->current_head->get_winrate(bestmove_idx);
		auto bestmove = global_tree->current_head->child_moves[bestmove_idx].value();

		// HACK
		//     : 本当は直接set すべきだけど、名前の衝突とかが怖かったり、他でも使ってる変数だから、
		//       他でもポインタとしてアクセスしないといけなくなる(アスタリスクを付ける)のでなんかきもい。
		// ポインタにセット
		if (ptr_bestmove != nullptr) { *ptr_bestmove = bestmove; }
		if (ptr_bestvalue != nullptr) { *ptr_bestvalue = bestvalue; }

		float cp;
		if (bestvalue == 1) {
			cp = 3e4;
		}
		else if (bestvalue == 0) {
			cp = -3e4;
		}
		else {
			// TODO: eval_coef
			cp = int(-logf(1.0 / bestvalue - 1.0) * 600);
		}

		std::string pv = __move_to_usi(bestmove);
		int depth = 1;
		UctNode* current_pv_node = global_tree->current_head;
		UctNode* next_pv_node = nullptr;    // current_pv_node での現状の最善手を指した時の、遷移先のnode
		while (current_pv_node->legal_moves_size > 0) {
			next_pv_node = current_pv_node->child_nodes[bestmove_idx].get();
			// 次のnode が無い
			// || 次のnode が未展開
			// || 現在のnode こそ評価されているものの、次のnode には一度も訪問していない。(TODO:この時、現在のnode の訪問数==1のはず... 確認しよう。)
			if (next_pv_node == nullptr
				|| next_pv_node->legal_moves_size <= 0
				|| next_pv_node->move_count == 0
				) {
				break;
			}
			current_pv_node = next_pv_node;    // 次のnode へ移動
			bestmove_idx = current_pv_node->get_bestmove_idx();
			pv += " " + __move_to_usi(current_pv_node->child_moves[bestmove_idx].value());
			++depth;    // NOTE: pv に手が付け加えられるタイミングは、depth が増えるタイミングである。
		}

		const auto& elapsed_time_ms = _stop_watch.elapsed_ms_interim();
		const auto& nps = (1000 * _playout_count) / elapsed_time_ms;

		// TODO
		//     : depth, nps
		//     : やねうら王の順番
		//       <2:info depth 1 seldepth 1 score cp 1054 nodes 55 nps 27500 time 2 pv 4e5d
		std::cout
			<< "info"
			<< " depth " << depth
			<< " score cp " << cp
			<< " nodes " << _playout_count
			<< " nps " << positive_decimal_to_int_str(nps)
			<< " time " << positive_decimal_to_int_str(elapsed_time_ms)
			<< " pv " << pv << std::endl;
	}

	// state info が壊れていないかをチェック。
	bool _check_state_info(__Board& board, std::string locate_info) {
		StateInfo* si = board.pos.get_state();
		for (int i = 0; i < board.ply() - 1; ++i) {
			if (!(si->previous != nullptr)) {
				std::cout << "Error: at " << locate_info << ", linked list of StateInfo is discontinuous." << std::endl;
				std::cout << "liked list can be traced until i = " << i << " (if i == 0, completely cannot trace.)" << std::endl;
				std::cout << "board.pos.get_state()->pliesFromNull = " << board.pos.get_state()->pliesFromNull << std::endl;
				std::cout << "si->pliesFromNull = " << si->pliesFromNull << std::endl;
				std::cout << "board.pos.get_start_state()->previous == nullptr = " << (board.pos.get_start_state()->previous == nullptr) << std::endl;
				std::cout << "ply = " << board.ply() << std::endl;
				std::cout << board.dump() << std::endl;
				throw std::runtime_error("Error");
			}
			if (!(si->pliesFromNull >= 0)) {
				std::cout << "Error: at " << locate_info << ", si->pliesFromNull < 0" << std::endl;
				std::cout << "liked list can be traced until i = " << i << " (if i == 0, completely cannot trace.)" << std::endl;
				std::cout << "board.pos.get_state()->pliesFromNull = " << board.pos.get_state()->pliesFromNull << std::endl;
				std::cout << "si->pliesFromNull = " << si->pliesFromNull << std::endl;
				std::cout << "board.pos.get_start_state()->previous == nullptr = " << (board.pos.get_start_state()->previous == nullptr) << std::endl;
				std::cout << "ply = " << board.ply() << std::endl;
				std::cout << board.dump() << std::endl;
				throw std::runtime_error("Error");
			}
			si = si->previous;
		}
	}
};

