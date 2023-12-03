#pragma once

#include <vector>
#include <mutex>
#include <atomic>
#include <memory>
#include <chrono>
#include <sstream>

#include "cshogi.h"
#include "util.hpp"

#include "debug_string_queue.hpp"
#include "stop_watch.hpp"

//#if defined(PARALLEL_PUCT_PLAYER)
#if 1

// TODO: static
constexpr int VALUE_NONE = -10000;
//constexpr int MOVE_COUNT_UNEVALED = -1;

// 対応するNode を確保していなくても、
// 展開済みの親Node さえ存在すればset 出来るように、
// Win, Lose, Draw はchildInfo にset する。
// state of UctChildInfo
constexpr uint8_t __STATE_WIN = 1 << 0;
constexpr uint8_t __STATE_LOSE = 1 << 1;
constexpr uint8_t __STATE_DRAW = 1 << 2;
constexpr uint8_t __STATE_NODE_EXIST = 1 << 3;    // 対応するnode の実体が存在している
// state of UctNode
constexpr uint8_t __STATE_EVALED = 1 << 0;
constexpr uint8_t __STATE_EXPANDED = 1 << 1;

// NOTE
//     : 新しくメンバ変数を増やしたら、コピーコンストラクタとか全部に反映すべし！！
typedef struct UctChildInfo {
	UctChildInfo()
		: move(Move::moveNone()), probability(0), move_count(0), sum_value(0), _state(0)
	{
	}

	// コピーコンストラクタ
	UctChildInfo(const UctChildInfo& obj) noexcept
		: move(obj.move), probability(obj.probability),
		  move_count(obj.move_count.load()), sum_value(obj.sum_value.load()), _state(obj._state.load())
	{
	}

	// (一旦普通で。。。)
	// 移動コンストラクタ
	UctChildInfo(UctChildInfo&& obj) noexcept
		: move(obj.move), probability(obj.probability),
		  move_count(obj.move_count.load()), sum_value(obj.sum_value.load()), _state(obj._state.load())
	{
	}

	// 代入演算子
	UctChildInfo& operator=(const UctChildInfo& obj) noexcept {
		move = obj.move;
		probability = obj.probability;
		move_count = obj.move_count.load();
		sum_value = obj.sum_value.load();
		_state = obj._state.load();
		return *this;
	}

	// 移動代入演算子
	UctChildInfo& operator=(UctChildInfo&& obj) noexcept {
		move = obj.move;
		probability = obj.probability;
		move_count = obj.move_count.load();
		sum_value = obj.sum_value.load();
		_state = obj._state.load();
		return *this;
	}

	inline void set_win() { _state |= __STATE_WIN; }
	inline void set_lose() { _state |= __STATE_LOSE; }
	inline void set_draw() { _state |= __STATE_DRAW; }
	inline void set_exist() { _state |= __STATE_NODE_EXIST; }
	inline bool is_win() const { return _state & __STATE_WIN; }
	inline bool is_lose() const { return _state & __STATE_LOSE; }
	inline bool is_draw() const { return _state & __STATE_DRAW; }
	inline bool is_exist() const { return _state & __STATE_NODE_EXIST; }
	
	// TODO: ここでもis_win とか考慮すべきかな。けど、これどこで使ってんかね。
	inline float get_winrate() const {
		if (move_count == 0) {
			return 0;
		}
		return sum_value / move_count;
	}

	// NOTE
	//     : この構造体には、エッジに関する情報(node が二つ存在しないと成立しない情報)を格納するべきかな？
	//       (二つのノードでそれぞれ持ってしまうと情報が重複すると思う & そもそもそれぞれで持つのって難しいかも)
	Move move;            // この子ノードへの指し手(エッジ)
	float probability;    // この子ノードへの遷移確率 by NN
	std::atomic<int> move_count;       // この子ノードへの訪問回数
	std::atomic<float> sum_value;      // この子ノードに遷移した際の"勝ち数の合計" (正確には合計報酬)


	// 0000 0001 ; 勝ち
	// 0000 0010 ; 負け
	// 0000 0100 ; 引き分け
	std::atomic<std::uint8_t> _state;
}uct_child_info_t;


// https://qiita.com/i153/items/38f9688a9c80b2cb7da7
// template<class T> int argmax_idx(const T& array, int size);
// 最大値を持つidx を返す。
inline int argmax_move_count_idx(const std::unique_ptr<uct_child_info_t[]>& array, int size) {
	auto crrt_max_element = array[0].move_count.load();
	int crrt_max_idx = 0;
	for (int i = 1; i < size; ++i) {
		const auto&& tmp = array[i].move_count.load();
		if (tmp > crrt_max_element) {
			crrt_max_element = tmp;
			crrt_max_idx = i;
		}
	}
	return crrt_max_idx;
}


// HACK: struct でええやん？
// TODO
//     : 名前もparallel node 的な名前に変更
// ParallelSearcher 用のnode。
class UctNode {
private:
	// 表面上配列を確保するだけで、中の本丸(UctNode, UctChildInfo)は確保しない。
	// 実行前
	//     - child_nodes: X, 要素のUctNode     : X
	//     - child_infos: X, 要素のUctChildInof: X
	// 実行後
	//     - child_nodes: O, 要素のUctNode     : X
	//     - child_infos: O, 要素のUctChildInof: X
	// その時々において適切なlegal_moves_size がset されている前提
	// 全てのunique_ptr を一度にまとめてmake_unique する。
	void _make_unique_at_once() {
		// NOTE: 長さlegal_moves_size で、要素が"UctNode へのunique_ptr" の配列
		child_nodes = std::make_unique<std::unique_ptr<UctNode>[]>(legal_moves_size);
		child_infos = std::make_unique<uct_child_info_t[]>(legal_moves_size);
	}

	// 実行前
	//     - child_nodes: O, i番目の要素のUctNode     : X
	//     - child_infos: O, i番目の要素のUctChildInof: X
	// 実行後
	//     - child_nodes: O, i番目の要素のUctNode     : X
	//     - child_infos: O, i番目の要素のUctChildInof: O
	void _init_single_node(int idx, Move move) {
		// TODO
		//     : これせんでも、自動でnullptr になるはず。。。
		child_nodes[idx] = nullptr;    // 本当に必要になるまで実体は持たない。
		auto& child_info = child_infos[idx];
		child_info.move = move;
		//child_info.move_count = 0;      // これせんでええよな
		//child_info.sum_value = 0;       // これせんでええよな
		//child_info.probability = 0;     // これせんでええよな
	}

	// 0000 0001 ; evaled
	// 0000 0010 : expanded
	std::atomic<std::uint8_t> _state;
public:
	// HACK: メンバをpublic で直接弄れる状態はやっぱりダメだね、手入れがしづらい。

	// 存在した瞬間から使用可能。
	std::atomic<int> move_count;             // ノードの訪問回数    (child_move_counts の合計)
	std::atomic<float> sum_value;            // 勝率の合計    (child_sum_values の合計)
	float value;                // 価値 (backup で使うぐらいで、そのあとは直接使われることは無い。)
	                            // 書き込みが発生するのは、評価される一度だけ。つまりこいつへのアクセスは実装上thread safe。
	std::atomic<float> sum_visited_probability;    // for fpu reduction

	// 展開後に使用可能。
	int legal_moves_size;    // HACK: child_num の方が適切かもね。

	// self のデストラクタが呼ばれると、以下の4つのunique_ptr に対してもデストラクタが呼ばれる感じかな？
	std::unique_ptr<std::unique_ptr<UctNode>[]> child_nodes;        // 子ノード達

	// TODO: こいつらは絶対にatomic にする必要がある
	// HACK: この4つ、一つの構造体に纏めた方がよいかもね。ChildInfo とかにさ。
	std::unique_ptr<uct_child_info_t[]> child_infos;


	UctNode() {
		_state = 0;
		move_count = 0;
		sum_value = 0;
		sum_visited_probability = 0;
		value = VALUE_NONE;    // TODO: これ必要か？
		legal_moves_size = -1;    // 0以上の時、ノードが展開されている。
	}

	// コピーコンストラクタ、定義してないからデフォルト移動コンストラクタ とかあるんじゃないかね？
	//UctNode& operator=(UctNode&& obj) noexcept {
	//	_evaled = obj._evaled;
	//	move_count = obj.move_count.load();
	//	sum_value = obj.sum_value.load();
	//	value = obj.value;
	//	legal_moves_size = obj.legal_moves_size;
	//	child_nodes = std::move(obj.child_nodes);
	//	child_infos = std::move(obj.child_infos);
	//}

	//UctNode(UctNode&& obj) noexcept
	//	: _evaled(obj._evaled), move_count(obj.move_count.load()), sum_value(obj.sum_value.load()), value(obj.value), legal_moves_size(obj.legal_moves_size),
	//	child_nodes(std::move(obj.child_nodes)), child_infos(std::move(obj.child_infos))
	//{
	//}

	// TODO: unique_ptr は確か自動で解放してくれたはず。。。一応調べよか。。。
	//~UctNode() {
	//}

	// NOTE: thread safe
	// 局面に評価が付いたら呼び出される。NN, 端手数の詰将棋, 引き分け 等、なんでも評価が付けばok。
	inline void set_evaled() { _state |= __STATE_EVALED; }
	//inline void set_expanded() { _state |= __STATE_EXPANDED; }
	inline bool is_evaled() const { return _state & __STATE_EVALED; }
	inline bool is_expanded() const { return _state & __STATE_EXPANDED; }
    
	// NOTE
	//     : probability をread しているので、eval_node() と並列に実行してはならない
	//     : legal_moves_size をread しているので、expand_node() と並列に実行してはならない。
	//     : 厳密にするなら、確率の和が1 か否かを確認しなければならないが、丸め誤差があるので面倒。
	//       0 なら全bit 0になっているはずで、"理論的に期待する値が0 -> 格納されている値も0" が成立するはず。
	//       なので、0 を用いて評価している。
	// is_evaled() だとしても、引き分けによる物なのか、NN によるものか分からない。
	// そこで、N によって推論して評価されたか否かを判定(厳密には必要条件)。
	inline bool is_evaled_by_nn() const {
		if (!this->is_evaled()) {    // evaled のグラフが立っていない奴は論外。
			return false;
		}

		for (int i = 0; i < legal_moves_size; ++i) {
			if (child_infos[i].probability > 0) {    // ひとつでも0超過 の値が格納されていれば、推論されているはず
				return true;
			}
		}
		// 全部0 なら推論されたとは言えない。
		return false;
	}

	// NOTE
	//     : legal_moves_size にアクセスする関数(当然この関数を含める)と並列してはならない。
	//     : 次の指し手(このノードにおける指し手)が議論に挙がるまでに展開されてる必要がある。
	// 指し手が0 の場合は別で検出して、ここまで来ないようにする(つもり)。
	// node の展開
	// (ひとまず、このような指し手があるということをすべて把握するが、実体は必要になるまで作成しない。)
	inline void expand_node(const __Board& board) {
		__LegalMoveList lm(board);
		legal_moves_size = lm.size();
		_make_unique_at_once();
		for (int i = 0; i < lm.size(); lm.next(), ++i) {
			auto lm_move = lm.Move();
			_init_single_node(i, lm_move);
		}
		// ここに直接置いたら、上記の操作全部write 扱いでメモリフェンスが働いてくれないだろうか。。。(希望的観測)
		_state |= __STATE_EXPANDED;
	}

	// HACK: create は微妙な希ガス。create_child_node() があるし、実際コイツはchild_node はcreate しない。
	//       せめてinit_single_child_node() とすべき。
	//       もっというと、expand_node_by_single_child() とかにすべき。
	// 既に展開されたnode で、再利用の際に呼ばれるだけなので、set_expanded() はしなくて良いはず。
	// 子ノードの実体は作成しない。
	// expand_node の、legal_move_size = 1 で特定の指し手だけ展開するversion とも言える。
	inline void init_and_create_single_child_node(Move move) {
		legal_moves_size = 1;
		_make_unique_at_once();
		_init_single_node(0, move);
	}

	// これを解さずにstd::make_unique<UctNode>() して良いのは探索木のroot node だけ。
	// 実行前
	//     - child_nodes: O, i番目の要素のUctNode     : X
	//     - child_infos: O, i番目の要素のUctChildInof: O
	// 実行後
	//     - child_nodes: O, i番目の要素のUctNode     : O
	//     - child_infos: O, i番目の要素のUctChildInof: O
	// 子ノードの実体作成。
	// expand_node() が呼ばれた後にのみ実行。
	inline UctNode* create_child_node(int idx) {
#ifdef DEBUG_
		if (!(legal_moves_size >= 0)) {    // 既に展開されている時。探索中はこちらになるはず。
			std::cout << "Error: at create_child_node()" << std::endl;
			exit(1);
		}
#endif
		child_nodes[idx] = std::make_unique<UctNode>();
		child_infos[idx].set_exist();

		// unique_ptr.get()すんのなんかきもいけどな。
		// まぁ、GCすんのは、こいつが用無しになってからやから問題は無い。
		return child_nodes[idx].get();
	}

	UctNode* release_children_except_one(Move move);

	Move get_move(const int idx) const {
		return child_infos[idx].move;
	}

	int get_move_count(const int idx) const {
		return child_infos[idx].move_count;
	}

	float get_winrate(const int idx) const {
		if (this->child_infos[idx].is_lose()) {
			return 1;
		}
		else if (this->child_infos[idx].is_win()) {
			return 0;
		}
		else if (child_infos[idx].move_count == 0) {
			return 0;
		}
		return child_infos[idx].sum_value / child_infos[idx].move_count;
	}

	//// TODO: win があるなら それを選択
	//// NOTE
	////     : legal_moves_size が確定した後(expand_node() が呼ばれた後) ならthread safe
	//inline int get_bestmove_idx() const {
	//	// NOTE: argmax_idx() はconst ObjTy& obj で参照してるだけなのでok。移動代入演算子は使ってない。(はず)
	//	return argmax_move_count_idx(child_infos, legal_moves_size);
	//}

	// expand_node() より後で実行されることを保証すること。
	//     : legal_moves_size でのデータ競合を回避。
	//       (まぁそもそも未展開だと上手く行かん)
	inline int get_bestmove_idx() const {
		//assert(global_tree->current_head->is_expanded());

		int crrt_max_move_count_exclude_lose = -1;
		int crrt_max_idx_exclude_lose = -1;
		int crrt_max_move_count = -1;
		int crrt_max_idx = -1;
		const int& total = this->legal_moves_size;

		int n_child_win_moves = 0;    // 子が勝つ指し手の数(親が負ける指し手の数)
		for (int i = 0; i < total; ++i) {
			const auto& crrt_child_info = this->child_infos[i];
			const auto child_move_count = crrt_child_info.move_count.load();

			if (crrt_child_info.is_lose()) {
				return i;
			}

			if (child_move_count > crrt_max_move_count) {
				crrt_max_move_count = child_move_count;
				crrt_max_idx = i;
			}
			if (crrt_child_info.is_win()) {    // 親が負け
				++n_child_win_moves;
			}
			else {
				if (child_move_count > crrt_max_move_count_exclude_lose) {
					crrt_max_move_count_exclude_lose = child_move_count;
					crrt_max_idx_exclude_lose = i;
				}
			}
		}

		if (n_child_win_moves == total) {
			// 負け確定なら、全ての手(負け確定の中でもマシな奴)から最善手を選択
			return crrt_max_idx;
		}
		else {
			// 負け確定してないなら、負け確定の手以外から選択。
			return crrt_max_idx_exclude_lose;
		}
	}

	inline Move get_bestmove() const {
		return child_infos[this->get_bestmove_idx()].move;
	}

	inline void print_child_move_counts() const {
		for (int i = 0; i < legal_moves_size; ++i) {
			std::cout << "[" << i << "] " << child_infos[i].move_count << std::endl;
		}
	}

	// HACK: これ、is_win, is_lose 考慮してないので非推奨
	// 訪問回数が多い順に並べられた、ノードのインデックスリストを返す
	// 軽くは無いので、頻繁に呼び出さないこと
	[[deprecated("This is because it does not take is_win and is_lose into account.")]]
	inline auto get_sorted_idx_list() const {
		auto&& ret = sorted_idx(child_infos, legal_moves_size, [](const uct_child_info_t& x, const uct_child_info_t& y) {
			return x.move_count > y.move_count;
			}
		);
		return ret;
	}

	// NOTE
	//     : legal_moves_size にアクセスしているので、expand_node より後で用いること。
	// (_should_exit_search() で使うことのみを考えているので、現状の実装でも良いはず。
	// 負け確定の中でも最善手と次善手のindex が欲しいならこの関数はこのままじゃ使えない。)
	// @return
	//     : 訪問回数が1番目, 2番目に大きい指し手のindex を返す。
	//     : 2番目に大きい指し手(次善手) が存在しなければ、{first_idx, -1} を返す。(他の指し手は全部親が負けとなる場合)
	//     : この局面が負け確定なら、{-1, -1} を返す。
	inline std::tuple<int, int> get_1th_2th_idx() const {
		int first_move_count = -1;
		int first_idx = -1;
		int second_move_count = -1;
		int second_idx = -1;

		const int& total = this->legal_moves_size;

		int n_child_win_moves = 0;    // 子が勝つ指し手の数(親が負ける指し手の数)
		for (int i = 0; i < total; ++i) {
			const auto& crrt_child_info = this->child_infos[i];
			const auto child_move_count = crrt_child_info.move_count.load();

			if (crrt_child_info.is_lose()) {
				// この手を指せば勝ち
				return {i, -1};
			}
			else if (crrt_child_info.is_win()) {    // 親が負け
				++n_child_win_moves;
				continue;
			}

			// 1番大きい奴が更新されたら、2番目も必然的に更新される
			if (child_move_count > first_move_count) {
				second_move_count = first_move_count;
				second_idx = first_idx;
				first_move_count = child_move_count;
				first_idx = i;
			}
			// 1番大きい訳じゃないけど、現状の2番よりは大きい
			else if (child_move_count > second_move_count) {
				second_move_count = child_move_count;
				second_idx = i;
			}
		}

		// 任意の子ノードで親が負ける場合、一度もfirst_idx, second_idx に代入されずに終わるので、初期値の-1, -1 が返る。
		return { first_idx, second_idx };
	}
};

class UctTree {
private:
	std::unique_ptr<UctNode> _gamebegin_node;
	Key _starting_pos_key;

	// @arg node: node.reset() した後に、current_head = node.get() することと等価
	void _reset_and_set_to_current(std::unique_ptr<UctNode>& node);

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
	[[deprecated("This is because it does not reuse subtree. Use reset_to_position() instead.")]]
	void reset() {
		_reset_and_set_to_current(_gamebegin_node);
		std::cout << "info string Warning: reset() is deprecated." << std::endl;
	}

	bool reset_to_position(Key new_starting_pos_key, const std::vector<Move>& moves) {
		// debug
		std::ostringstream oss;
		stopwatch_t __sw1;
		stopwatch_t __sw2;
		stopwatch_t __sw3;
		clear_debug_str();


		__sw1.start();
		if (_starting_pos_key != new_starting_pos_key) {
			_reset_and_set_to_current(_gamebegin_node);
			sync_cout << "info string got different starting pos" << sync_endl;
		}

		_starting_pos_key = new_starting_pos_key;

		UctNode* old_head = current_head;
		UctNode* prev_head = nullptr;    // 棋譜を辿る際に、現在の一つ手前のnode を指す。
		current_head = _gamebegin_node.get();    // 開始地点から指し進めていく。
		bool seen_old_head = old_head == current_head;
		__sw1.stop();

		__sw2.start();
		// 棋譜を辿る
		for (const auto& move : moves) {
			//std::cout << "info string reset_to_position(): move = " << __move_to_usi(move.value()) << std::endl;
			prev_head = current_head;
			current_head = current_head->release_children_except_one(move);
			if (old_head == current_head) {
				seen_old_head = true;
			}
		}
		__sw2.stop();

		__sw3.start();
		// この時点で、current_head は、新しくset された局面に相当するnode を指し示す。
		// old_head も見つからず、同じ局面がset された訳でもない時、再利用不可
		if (!seen_old_head && current_head != old_head) {
			if (prev_head != nullptr) {    // 今の局面は開始局面ではない
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
		__sw3.stop();

		// debug
		oss
			<< "moves.size()=" << moves.size()
			<< ", reset_to_position(){time1="
			<< __sw1.elapsed_ms() << ",time2=" << __sw2.elapsed_ms() << ",time3=" << __sw3.elapsed_ms()
			<< "}";
		enqueue_debug_str(oss.str());

		return seen_old_head;
	}

};

// probabilitys に格納されたロジットを、確率に変換。
inline void softmax_with_temperature(UctNode* node, const float temp) {
	// 0 にしておくことで、0以上の大きい数字が一つもない時、
	// overflow 対策は発生しない。
	float max_logit = 0;

	// DEBUG: 一旦dlshogi に合わせる。
	const float temp_inverse = 1 / temp;

	// 温度
	for (int i = 0; i < node->legal_moves_size; ++i) {
		float& logit = node->child_infos[i].probability;
		// TODO
		//     : 確か割り算は重かったので、temp が定数故掛け算用の定数をisready で計算して掛け算に変換すべきかな。
		
		// DEBUG: 一旦dlshogi に合わせる。
		//logit /= temp;
		logit *= temp_inverse;

		if (logit > max_logit) {
			max_logit = logit;
		}
	}

	// オーバーフロー対策で最大値を引く
	float sum = 0;
	for (int i = 0; i < node->legal_moves_size; ++i) {
		float& logit = node->child_infos[i].probability;

		logit = expf(logit - max_logit);
		sum += logit;
	}

	// 正規化(softmax)
	for (int i = 0; i < node->legal_moves_size; ++i) {
		float& logit = node->child_infos[i].probability;

		logit /= sum;
	}
}

//extern std::unique_ptr<UctTree> global_tree;

#endif