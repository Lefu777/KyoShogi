#pragma once

#include "cshogi.h"
#include "util.hpp"
#include <vector>
#include <memory>

#if defined(RANDOM_PLAYER) || defined(UCT_RANDOM_PLAYER) || defined(PUCT_PLAYER)

// TODO: static
constexpr int VALUE_NONE = -10000;

// HACK: struct でええやん？
class UctNode {
private:
	// その時々において適切なlegal_moves_size がset されている前提
	// 全てのunique_ptr を一度にまとめてmake_unique する。
	void _make_unique_at_once() {
		child_nodes = std::make_unique<std::unique_ptr<UctNode>[]>(legal_moves_size);
		child_moves = std::make_unique<Move[]>(legal_moves_size);
		child_move_counts = std::make_unique<int[]>(legal_moves_size);
		child_sum_values = std::make_unique<float[]>(legal_moves_size);
		child_probabilitys = std::make_unique<float[]>(legal_moves_size);
	}

	void _init_single_node(int idx, Move move) {
		child_nodes[idx] = nullptr;    // 本当に必要になるまで実体は持たない。
		child_moves[idx] = move;
		child_move_counts[idx] = 0;
		child_sum_values[idx] = 0;
		child_probabilitys[idx] = 1;
	}

	bool _evaled;                // この局面について推論したらtrue に。

public:
	// 存在した瞬間から使用可能。
	int move_count;             // ノードの訪問回数    (child_move_counts の合計)
	float sum_value;            // 勝率の合計    (child_sum_values の合計)
	float value;                // 価値

	// 展開後に使用可能。
	int legal_moves_size;    // HACK: child_num の方が適切かもね。

	// self のデストラクタが呼ばれると、以下の4つのunique_ptr に対してもデストラクタが呼ばれる感じかな？
	std::unique_ptr<std::unique_ptr<UctNode>[]> child_nodes;        // 子ノード達

	// TODO: こいつらは絶対にatomic にする必要がある
	// HACK: この4つ、一つの構造体に纏めた方がよいかもね。ChildInfo とかにさ。
	std::unique_ptr<Move[]> child_moves;            // 子ノードへの指し手(エッジ)
	std::unique_ptr<int[]> child_move_counts;      // 子ノードへの訪問回数
	std::unique_ptr<float[]> child_sum_values;     // 子ノードに遷移した際の"勝ち数の合計"
	                                               // update_result() により更新される。
	std::unique_ptr<float[]> child_probabilitys;   // 子ノードへの遷移確率 by NN


	UctNode() {
		move_count = 0;
		sum_value = 0;
		value = VALUE_NONE;
		legal_moves_size = -1;    // 0以上の時、ノードが展開されている。
		_evaled = false;
	}

	~UctNode() {
		// HACK
		//     : 恐らくこの括弧はなくても、このデストラクタのスコープ抜ける時のGCされるから要らんはず。
		//       -> 恐らくというか、ローカル変数だから当たり前。まぁでも念の為 of 念の為で一旦放置。
		{
			auto temp1 = std::move(child_nodes);
			auto temp2 = std::move(child_moves);
			auto temp3 = std::move(child_move_counts);
			auto temp4 = std::move(child_sum_values);
			auto temp5 = std::move(child_probabilitys);
		}
	}

	void set_evaled() { _evaled = true; }
	bool is_evaled() const { return _evaled; }

	// NOTE
	//     : 次の指し手(このノードにおける指し手)が議論に挙がるまでに展開されてる必要がある。
	// 指し手が0 の場合は別で検出して、ここまで来ないようにする(つもり)。
	// node の展開
	// (ひとまず、このような指し手があるということをすべて把握するが、実体は必要になるまで作成しない。)
	void expand_node(const __Board& board) {
		__LegalMoveList lm(board);
		legal_moves_size = lm.size();
		_make_unique_at_once();
		for (int i = 0; i < lm.size(); lm.next(), ++i) {
			auto lm_move = lm.Move();
			_init_single_node(i, lm_move);
		}
	}

	// 子ノードの実体は作成しない。
	// expand_node の、legal_move_size = 1 で特定の指し手だけ展開するversion とも言える。
	void init_and_create_single_child_node(Move move) {
		legal_moves_size = 1;
		_make_unique_at_once();
		_init_single_node(0, move);
	}

	// 子ノードの実体作成。
	// expand_node() が呼ばれた後にのみ実行。
	UctNode* create_child_node(int idx) {
#ifdef DEBUG
		if (!(legal_moves_size >= 0)) {    // 既に展開されている時。探索中はこちらになるはず。
			std::cout << "Error: at create_child_node()" << std::endl;
			exit(1);
		}
#endif
		child_nodes[idx] = std::make_unique<UctNode>();
		// unique_ptr.get()すんのなんかきもいけどな。
		// まぁ、GCすんのは、こいつが用無しになってからやから問題は無い。
		return child_nodes[idx].get();
	}
	// @arg move: この指し手に対応する子ノード以外を全て解放する。
	UctNode* release_children_except_one(Move move) {
		if (legal_moves_size >= 0) {
			bool found = false;
			// HACK: 参照渡しした方が速い。
			for (int i = 0; i < legal_moves_size; ++i) {
				if (child_moves[i] == move) {    // 発見。
					found = true;
					if (!child_nodes[i]) {    // 実体が無ければ作る。
						//std::cout << "info string rceo(): meet nullptr start" << std::endl;
						child_nodes[i] = std::make_unique<UctNode>();
						//std::cout << "info string rceo(): meet nullptr done" << std::endl;
					}
					else {
#ifdef DEBUG_
						std::cout << "info string rceo(): exist" << std::endl;
						std::cout << "info string rceo(): child_move_counts[i] = " << child_move_counts[i] << std::endl;
						std::cout << "info string rceo(): move_count = " << move_count << std::endl;
#endif
#ifdef DEBUG
						std::cout << "info string reuse subtree ["
							<< child_move_counts[i] << " / " << move_count << "]" << std::endl;
#endif
					}
					// else { // 実体がすでに有れば何もせずにそれを再利用。 }

					// 残すnode を0番目に持ってくる
					if (i != 0) {    // 0番目でない時
						// NOTE
						//     : 他の要素は入れ替えてないけど、使わんやろ
						//    -> 後そもそも、0番目のポインタは解放済みなので、このポインタは保持しなくても良い。
						child_nodes[0] = std::move(child_nodes[i]);
						child_moves[0] = child_moves[i];
						child_move_counts[0] = child_move_counts[i];
						child_sum_values[0] = child_sum_values[i];
						child_probabilitys[0] = child_probabilitys[i];
					}
				}
				else {
					//std::cout << "info string rceo(): got = " << __move_to_usi(child_moves[i]) << std::endl;
					// TODO: if (child_nodes[i]) に。
					if (child_nodes[i]) {
						{
							// GC
							auto temp = std::move(child_nodes[i]);
						}
						// TODO: 別thread で動くGC
					}
				}
			}
			if (found) {
				legal_moves_size = 1;
				return child_nodes[0].get();
			}
			else {
#ifdef DEBUG_
				std::cout << "info string not found subtree, move = " << __move_to_usi(move.value()) << std::endl;
#endif
				init_and_create_single_child_node(move);
				// 実体を作成して、それを返す。
				// HACK: return create_child_node(0);
				return (child_nodes[0] = std::make_unique<UctNode>()).get();
			}
		}
		else {
#ifdef DEBUG_
			std::cout << "info string not found subtree, because this is unexpanded node, move = " << __move_to_usi(move.value()) << std::endl;
#endif
			init_and_create_single_child_node(move);
			// 実体を作成して、それを返す。
			return (child_nodes[0] = std::make_unique<UctNode>()).get();
		}
	}

	Move get_move(const int idx) const {
		return child_moves[idx];
	}

	int get_move_count(const int idx) const {
		return child_move_counts[idx];
	}

	float get_winrate(const int idx) const {
		return child_sum_values[idx] / child_move_counts[idx];
	}

	int get_bestmove_idx() const {
		// NOTE: argmax_idx() はconst ObjTy& obj で参照してるだけなのでok。移動代入演算子は使ってない。(はず)
		return argmax_idx(child_move_counts, legal_moves_size);
	}

	Move get_bestmove() const {
		return child_moves[this->get_bestmove_idx()];
	}

	void print_child_move_counts() const {
		for (int i = 0; i < legal_moves_size; ++i) {
			std::cout << "[" << i << "] " << child_move_counts[i] << std::endl;
		}
	}

	// 訪問回数が多い順に並べられた、ノードのインデックスリストを返す
    // 軽くは無いので、頻繁に呼び出さないこと
	auto get_sorted_idx_list() const {
		auto&& ret = sorted_idx_descending(child_move_counts, legal_moves_size);
		return ret;
	}

};

class UctTree {
private:
	std::unique_ptr<UctNode> _gamebegin_node;
	Key _starting_pos_key;

	// @arg node: node.reset() した後に、current_head = node.get() することと等価
	void _reset_and_set_to_current(std::unique_ptr<UctNode>& node) {
		if (node) {
			auto temp = std::move(node);
		}
		node = std::make_unique<UctNode>();
		current_head = node.get();
	}

public:
	// NOTE
	//     : ポインタなので値が弄れるので、"private において、getter だけ用意して、
	//       外部からは値を見れるけど、弄れない" みたいなことが出来ない。
	UctNode* current_head;

	UctTree() {
		_starting_pos_key = 0;
		_reset_and_set_to_current(_gamebegin_node);
	}

	// TODO
	//     : 不要になったTree のデストラクト
	//     : 可能ならば別のthread でGC
	//     : ノードの再利用。
	// 非推奨
	void reset() {
		_reset_and_set_to_current(_gamebegin_node);
		std::cout << "info string Warning: reset() is deprecated." << std::endl;
	}

	bool reset_to_position(Key new_starting_pos_key, std::vector<Move> moves) {
		if (_starting_pos_key != new_starting_pos_key) {
			_reset_and_set_to_current(_gamebegin_node);
			std::cout << "info string got different starting pos" << std::endl;
		}

		_starting_pos_key = new_starting_pos_key;

		UctNode* old_head = current_head;
		UctNode* prev_head = nullptr;    // 棋譜を辿る際に、現在の一つ手前のnode を指す。
		current_head = _gamebegin_node.get();    // 開始地点から指し進めていく。
		bool seen_old_head = old_head == current_head;

		// 棋譜を辿る
		for (const auto& move : moves) {
			//std::cout << "info string reset_to_position(): move = " << __move_to_usi(move.value()) << std::endl;
			prev_head = current_head;
			current_head = current_head->release_children_except_one(move);
			if (old_head == current_head) {
				seen_old_head = true;
			}
		}

		// TODO: 見つからなかった時
		// この時点で、current_head は、新しくset された局面に相当するnode を指し示す。
		// old_head も見つからず、同じ局面がset された訳でもない時、再利用不可
		if (!seen_old_head && current_head != old_head) {
			if (prev_head != nullptr) {    // 開始局面ではない
				// 以前に探索した局面の祖先である可能性がある。
				// この時、current_head よりも進んだ先で、old_head がある可能性がある。
				// なので、そいつらを纏めて解放する。
				// TODO
				//     : 2, 3手先でold_head があるなら、暫くしたらそのnode に辿り着くはずで、
				//       メモリ余裕があるなら残してもええと思うけど、通常の実践じゃ、そもそもさかのぼって、
				//       中途半端な局面がset されることはないから、まぁ一旦これで。
				_reset_and_set_to_current(prev_head->child_nodes[0]);
			}
			else {
				_reset_and_set_to_current(_gamebegin_node);
			}
		}
		return seen_old_head;
	}

};

// probabilitys に格納されたロジットを、確率に変換。
inline void softmax_with_temperature(UctNode* node, const float temp) {
	// 0 にしておくことで、0以上の大きい数字が一つもない時、
	// overflow 対策は発生しない。
	float max_logit = 0;

	// 温度
	for (int i = 0; i < node->legal_moves_size; ++i) {
		float& logit = node->child_probabilitys[i];
		logit *= temp;
		if (logit > max_logit) {
			max_logit = logit;
		}
	}

	// オーバーフロー対策で最大値を引く
	float sum = 0;
	for (int i = 0; i < node->legal_moves_size; ++i) {
		float& logit = node->child_probabilitys[i];

		logit = expf(logit - max_logit);
		sum += logit;
	}

	// 正規化(softmax)
	for (int i = 0; i < node->legal_moves_size; ++i) {
		float& logit = node->child_probabilitys[i];

		logit /= sum;
	}
}

// HACK: ここに置くのは正直微妙。例えば、教師生成の時に、二つのglobal_tree を持ちたいってなったら微妙。他にもいろいろ、兎に角微妙だよね。
extern std::unique_ptr<UctTree> global_tree;

#endif