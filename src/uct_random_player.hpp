#pragma once

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>

#include "uct_node.hpp"
#include "base_player.hpp"
#include "random_player.hpp"
#include "util.hpp"
#include "config.hpp"

// TODO
//     : まずは、class のメンバ関数をthreadで並列実行して、
//       global変数 or static メンバ変数を正しくインクリメントする、というテストをしよう。
//       (staticメンバ変数は厳しそう & staticメンバ変数の使い方一通り調べたい & dlshogiはglobal変数使ってるっぽい。)

//extern std::unique_ptr<UctTree> global_tree;
//extern UctTree* global_tree;

#ifdef UCT_RANDOM_PLAYER

typedef std::pair<UctNode*, int> UctNodeAndMove;
typedef std::pair<UctNode*, Color> UctNodeAndColor;
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
constexpr int RESULT_QUEUING = -1;
constexpr int RESULT_DISCARDED = -2;


// TODO
//     : 抽象クラスというか、基底クラスを作る。
//       ほんでもって、override 的なことして、基底のPlayerとしてオブジェクトは持ってるけど、
//       random_player のを呼び出す、的なのをしたい。
//     : なんか、header の中身変えただけだと、.exe に反映されん。

class UctRandomPlayer : public BasePlayer {
private:
	__Board _board;

	std::string _position_str;
	unsigned long long _playout_count;
	//UctTree _tree;
	//UctTree* global_tree;
	//static UctTree* global_tree;
	std::vector<__Board> _boards;    // leaf node からのplayout における開始局面
	std::vector<UctNodeAndColor> _eval_queue;
	int _last_pv_print_playout;

	// USI option
	int _batch_size;
	int _const_playout;
	int _leaf_const_playout;
	int _max_ply_from_leaf;    // leaf node からのplayout における最大手数
	int _print_pv_per_playout;

public:
	UctRandomPlayer() : BasePlayer() {
		_board = __Board();
		//global_tree.reset(new UctTree());
		global_tree = std::make_unique<UctTree>();
		//global_tree = new UctTree();
		_position_str = "";
		_playout_count = 0;
		_last_pv_print_playout = 0;

		_batch_size = 10;    // TODO: default option をどこかに。。。
		_const_playout = 10000;
		_leaf_const_playout = 1;
		_max_ply_from_leaf = 512;
		_print_pv_per_playout = 1000;    // TODO: 時間制御諸々
	}

	~UctRandomPlayer() {

	}

	void usi() {
		std::cout << "id name UctRandomPlayerV0.02.8" << std::endl;
		std::cout << "id author lefu777" << std::endl;
		std::cout << "option name " << "USI_Ponder" << " type " << "check" << " default " << (false ? "true" : "false") << std::endl;
		std::cout << "option name " << "batch_size" << " type " << "spin" << " default " << _batch_size << " min " << 0 << " max " << INT_MAX << std::endl;
		std::cout << "option name " << "const_playout" << " type " << "spin" << " default " << _const_playout << " min " << 0 << " max " << INT_MAX << std::endl;
		std::cout << "option name " << "leaf_const_playout" << " type " << "spin" << " default " << _leaf_const_playout << " min " << 0 << " max " << INT_MAX << std::endl;
		std::cout << "option name " << "max_ply_from_leaf" << " type " << "spin" << " default " << _max_ply_from_leaf << " min " << 0 << " max " << INT_MAX << std::endl;
		std::cout << "option name " << "print_pv_per_playout" << " type " << "spin" << " default " << _print_pv_per_playout << " min " << 0 << " max " << INT_MAX << std::endl;
		std::cout << "usiok" << std::endl;
	}

	// @arg name: name <id> value <x>
	void setoption(std::string cmd) {
		auto tokens = tokenize(cmd);
		if (tokens[1] == "batch_size") {
			_batch_size = std::atoi(tokens[3].c_str());
		}
		else if (tokens[1] == "const_playout") {
			_const_playout = std::atoi(tokens[3].c_str());
		}
		else if (tokens[1] == "leaf_const_playout") {
			_leaf_const_playout = std::atoi(tokens[3].c_str());
		}
		else if (tokens[1] == "max_ply_from_leaf") {
			_max_ply_from_leaf = std::atoi(tokens[3].c_str());
		}
		else if (tokens[1] == "print_pv_per_playout") {
			_print_pv_per_playout = std::atoi(tokens[3].c_str());
		}
		else {
			std::cout << "Warning: unexpected setoption name = " << tokens[1] << std::endl;
		}
	}

	// TODO: 本当は、ここでinit とか呼ぶ。
	bool _isready() {
		_board.reset();
		//global_tree->reset();
		_boards.reserve(_batch_size);
		return true;
	}

	void isready() {
		if (_isready()) {
			std::cout << "readyok" << std::endl;
		}
		else {
			std::cout << "Error: isready failed" << std::endl;
			exit(1);
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
		std::string startsfen;
		std::vector<Move> moves;
		Key startpos_key;
		_board.set_position_and_get_components(_position_str, &startsfen, &moves, &startpos_key);
		bool found = global_tree->reset_to_position(startpos_key, moves);
		std::cout << "info string found_old_head = "
			<< (found ? "true" : "false") << std::endl;

		const auto& mate_move3 = _board.mateMove(3);
		std::cout << "info strig mate_move3 = " << mate_move3 << std::endl;

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
		}
	}

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
				exit(1);
			}
			if (!(si->pliesFromNull >= 0)) {
				std::cout << "Error: at " << locate_info << ", si->pliesFromNull < 0" << std::endl;
				std::cout << "liked list can be traced until i = " << i << " (if i == 0, completely cannot trace.)" << std::endl;
				std::cout << "board.pos.get_state()->pliesFromNull = " << board.pos.get_state()->pliesFromNull << std::endl;
				std::cout << "si->pliesFromNull = " << si->pliesFromNull << std::endl;
				std::cout << "board.pos.get_start_state()->previous == nullptr = " << (board.pos.get_start_state()->previous == nullptr) << std::endl;
				std::cout << "ply = " << board.ply() << std::endl;
				std::cout << board.dump() << std::endl;
				exit(1);
			}
			si = si->previous;
		}
	}

	// https://github.com/TadaoYamaoka/python-dlshogi/blob/master/pydlshogi/player/mcts_player.py
	// 終局 or 宣言勝ち の局面では呼び出さない。
	int _go() {

		// _check_state_info(_board, "_go()");

		assert(!(_board.is_game_over() || _board.is_nyugyoku()));

		// プレイアウト数をクリア
		_playout_count = 0;
		_last_pv_print_playout = 0;

		// 候補手が一つの場合は、その手を返す。
		__LegalMoveList legal_moves(_board);
		if (legal_moves.size() == 1) {
			return legal_moves.move();
		}

#ifdef DEBUG
		std::cout << "info string debug: search() start" << std::endl;
#endif
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
		// NOTE
		//     : uct_search() で、一旦全部trajectories_batch に入る。
		//     : - QUEUING はtrajectories_batch へ。推論後にbackup する為。
		//       - DISCARDED はtrajectories_batch_discarded へ。VIRTUAL_LOSS を元に戻すため。
		//       - それ以外(勝敗が明確に分かった場合) はどこにも残らない。
		std::vector<Trajectory> trajectories_batch;
		std::vector<Trajectory> trajectories_batch_discarded;
		int current_batch_top;

		while (1) {
			trajectories_batch.clear();
			trajectories_batch_discarded.clear();
			current_batch_top = -1;    // top の要素のidx
			std::vector<__Board> board_cpys(_batch_size, _board);
			for (int i = 0; i < _batch_size; ++i) {
				// TODO: これ、本当はboard は参照渡しして、uct_search() でundoMove() (というより、pop())もするようにすべき。
				// 明示的なコピーコンストラクタの呼出し
#ifdef DEBUG_
				std::cout << "info: start copy constructer" << std::endl;
#endif
				//_check_state_info(_board, "search() (before cpy)");	
				// NOTE: やっぱこれ、local だったのが良くなかったんやで。
				//__Board board_cpy(_board);    // forの末尾で消える。
				__Board& board_cpy = board_cpys[i];
				// _check_state_info(board_cpy, "search() (after cpy)");
#ifdef DEBUG_
				std::cout << "info: end copy constructer" << std::endl;
#endif
				// TODO: これで本当にからのtrajectory 入ってるかは怪しい。
				trajectories_batch.emplace_back(std::initializer_list<UctNodeAndMove>{});
				++current_batch_top;

				int result = uct_search(board_cpy, global_tree->current_head, trajectories_batch[current_batch_top]);

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
			if (trajectories_batch.size() > 0) {
				eval_node();
			}

			// Virtual Loss を元に戻す
			for (const auto& trajectory : trajectories_batch_discarded) {
				for (const auto& nm : trajectory) {
					nm.first->move_count -= VIRTUAL_LOSS;
					nm.first->child_move_counts[nm.second] -= VIRTUAL_LOSS;
				}
			}

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
						// 子ノードの手番からのvalue なので反転する。
						result = 1 - node->child_nodes[next_index]->value;
					}
					update_result(node, next_index, result);
					result = 1 - result;    // NOTE: 手番に応じて反転？(@20230819)
				}
			}

			// 探索を打ち切るか確認
			if (_playout_count >= _const_playout) {
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
	int uct_search(__Board& board, UctNode* current_node, Trajectory& trajectory) {
#ifdef DEBUG_
		std::cout << "info debug: uct_search() start" << std::endl;
#endif
		if (current_node->legal_moves_size < 0) {    // 未展開
			current_node->expand_node(board);
		}
		int next_index = select_max_ucb_child(current_node);
		board.push(current_node->child_moves[next_index].value());

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
		// TODO
		//     : 本当はVALUE_NONE とかで初期化すべきだけど、uct_node の奴を使うのなんかきもくねぇか。
		//    -> どっかに纏めたい。constant.hpp/.cpp として。
		int result = VALUE_NONE;
		if (current_node->child_nodes[next_index] == nullptr) {    // child_nodes[next_index]の実体がない -> 実体を作成して評価
#ifdef DEBUG_
			std::cout << "info debug: uct_search() child_node[next_idx] == nullptr" << std::endl;
#endif
			UctNode* child_node = current_node->create_child_node(next_index);

			// child_node の局面において引き分けをチェック。
			auto repetition_type = board.isDraw(INT_MAX);
			if (repetition_type == NotRepetition) {
				if (board.is_nyugyoku() || board.mateMove(3)) {
					child_node->value = VALUE_WIN;
					result = 0;
				}
				else {
					// 以下は同じ意味。
#if 1
					child_node->expand_node(board);
					if (child_node->legal_moves_size == 0) {
#else
					if (board.is_game_over()) {
#endif
						child_node->value = VALUE_LOSE;
						result = 1;
					}
					else {
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
				switch (repetition_type) {
				case RepetitionDraw:
					child_node->value = VALUE_DRAW;
					result = 0.5;
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
					exit(1);
				}
			}
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
				std::cout << "info Warning: at uct_search(), next_node->value != VALUE_LOSE, but next_node->legal_moves_size == 0" << std::endl;
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
#ifdef DEBUG
		if (result == RESULT_QUEUING || result == RESULT_DISCARDED) {
			std::cout << "info Warning: at uct_search(), although not after recursion, the result is RESULT_QUEUING or RESULT_DISCARDED" << std::endl;
			return result;
		}
#endif
		// NOTE
		//     : queueing ではない場合、backup(探索木への探索結果の反映) はここで行う。
		//       (backup は、queueing の場合はeval_node()後に行い、
		//       discarded の場合はそもそもbackup は無理だしする必要もなく、それ以外の勝敗が決している場合はここで行う。)
		update_result(current_node, next_index, result);

		return 1 - result;
	}

	// 一番探索すべき指し手のindex を返す。
	int select_max_ucb_child(UctNode* node) const {
		int total = node->legal_moves_size;
#ifdef DEBUG_
		assert(total > 0 && "Warning: this node may be unexpanded.");
		std::cout << (node == nullptr ? "nullptr" : "not null") << std::endl;
		std::cout << node->legal_moves_size << std::endl;
#endif
		std::vector<int> current_max_idx;
		float reward_mean, bonus;
		float current_max = -1;    // スコアが負にならない前提。
		for (int i = 0; i < total; ++i) {
			if (node->child_move_counts[i] == 0) {
				reward_mean = 0;
			}
			else {
				reward_mean = node->child_sum_values[i] / node->child_move_counts[i];
			}
			if (node->move_count == 0) {
				// log0 の回避
				bonus = 0;
			}
			else {
				bonus = sqrt((2 * logf(node->move_count)) / (1 + node->child_move_counts[i]));
			}
			if (reward_mean + bonus > current_max) {
				current_max = reward_mean + bonus;
				current_max_idx.clear();
				current_max_idx.emplace_back(i);
			}
			else if (reward_mean + bonus == current_max) {
				current_max_idx.emplace_back(i);
			}
#ifdef DEBUG_
			std::cout << reward_mean << ", " << bonus << std::endl;;
			std::cout << current_max_idx << ", " << current_max << std::endl;;
#endif
		}

#ifdef DEBUG_
		std::cout <<"final = " << current_max_idx << ", " << current_max << std::endl;;
#endif

		return current_max_idx[mt_genrand_int32() % current_max_idx.size()];
	}

	void update_result(UctNode* node, int next_index, float result) {
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
	void queue_node(UctNode* node, __Board& board) {
		// _check_state_info(board, "queue_node() (ref arg)");
		_boards.emplace_back(board);    // board のコピーを渡す
#ifdef DEBUG
		std::ostringstream oss;
		oss << "queue_node() (_boards[_boards.size() - 1], _boards.size() = " << _boards.size() << ")";
		// _check_state_info(_boards[_boards.size() - 1], oss.str());
#endif
		_eval_queue.emplace_back(node, board.turn());
	}

	// queue の中身を一斉に評価
	void eval_node() {
		std::vector<float> values_mean(_boards.size(), 0);
		
		for (int i = 0; i < _boards.size(); ++i) {
			__Board& board = _boards[i];
			// _check_state_info(board, "eval_node() (__Board& board = _boards[i])");
			for (int j = 1; j <= _leaf_const_playout; ++j) {
				float result = play_once(board, _eval_queue[i].second);
				values_mean[i] += (result - values_mean[i]) / j;
#ifdef DEBUG_
				std::cout << "result[" << j << "]: " << result << std::endl;
#endif
			}
			// _check_state_info(board, "eval_node() (after playout");
#ifdef DEBUG_
			std::cout << "values_mean[" << i << "]: " << values_mean[i] << std::endl;
#endif
		}

		// TODO: _eval_queue を走査。
		for (int i = 0; i < _boards.size(); ++i) {
			_eval_queue[i].first->value = values_mean[i];
#ifdef DEBUG_
			std::cout << "updated value = " << _eval_queue[i].first->value << std::endl;
#endif
		}

		_eval_queue.clear();
		_boards.clear();
	}

	// @arg board: node 
	// @arg turn: 最後に返すどちら側からみたvalue(result) を返すか。
	float play_once(__Board& board, Color turn) {
		// _check_state_info(board, "play_once() (ref arg)");
#ifdef DEBUG_
		std::cout << "play_once() start" << std::endl;
#endif
		float result = 0.5;
		int ply = 0;
		for (; ply < _max_ply_from_leaf; ++ply) {
#ifdef DEBUG_
			std::cout << "play_once() ply = " << ply << std::endl;
#endif
			// _check_state_info(board, "play_once() (at loop)");
			if (board.is_game_over()) {
				result = board.turn() == turn ? 0 : 1;
				break;
			}
			else if (board.is_nyugyoku() || board.mateMove(3)) {
				result = board.turn() == turn ? 1 : 0;
				break;
			}
#ifdef DEBUG_
			std::cout << "info: is_game_over(), is_nyugyoku(), mateMove() done " << std::endl;
#endif
			auto repetition_type = board.isDraw(INT_MAX);
#ifdef DEBUG_
			std::cout << "info: get repetition_type" << std::endl;
#endif
			if (repetition_type == RepetitionDraw) {
				result = 0.5;
				break;
			}
			// TODO: random playoutにおいて以下で勝ち/負け にするのは微妙な気もする。
			else if (repetition_type == RepetitionWin || repetition_type == RepetitionSuperior) {
				result = board.turn() == turn ? 1 : 0;
				break;
			}
			else if (repetition_type == RepetitionLose || repetition_type == RepetitionInferior) {
				result = board.turn() == turn ? 0 : 1;
				break;
			}
#ifdef DEBUG_
			else{
				std::cout << "info: repetition_type == NotRepetition" << std::endl;
				assert(repetition_type == NotRepetition);
			}
#endif

#ifdef DEBUG_
			std::cout << "info: repetition_type done " << std::endl;
#endif

			board.push(RandomPlayer::_get_random_move(board));
		}
		// undoMove()
		for (; ply > 0; --ply) {
			board.pop();
		}
		return result;    // 最大手数により引き分け。
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

		// TODO
		//     : depth, nps
		//     : やねうら王の順番
		//       <2:info depth 1 seldepth 1 score cp 1054 nodes 55 nps 27500 time 2 pv 4e5d
		std::cout << "info"
			<< " depth " << depth << " nodes " << _playout_count << " score cp " << cp
			<< " nps " << 0 << " time " << 0 << " pv " << pv << std::endl;
	}
};

#endif