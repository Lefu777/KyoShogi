#pragma once

#include "parallel_puct_player.hpp"
#include "debug_string_queue.hpp"
#include "pld_compiler.hpp"

//#ifdef PARALLEL_PUCT_PLAYER
#if 1

// ==================================================
// 定数
// ==================================================

//// 探索時定数

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

//// 探索パラメータ

//constexpr int puct_c_base = 19652;
//constexpr float puct_c_init = 1.25;

// HACK; 50, 25 ぐらいの方が良いと思うけど、どうやろうか。
//// 持ち時間制御に使用

//constexpr int div_init = 54;    // 初期値(最大値)
//constexpr int div_second = 50;
//constexpr int div_final = 14;    // 最小値
//constexpr int n_self_ply_while_begining = (div_init - div_second) / 2;    // 序盤に自分が指す手の数

//// その他

// must be 2^n
constexpr uint64_t POS_MUTEX_NUM = 65536;
// monitoring thread となるthread のid (GPUが1つの場合のみを想定)(これ正直要らんね)
constexpr int MONITORING_THREAD_ID = 0;

const std::string MOVE_USI_RESIGN = "resign";
const std::string MOVE_USI_WIN = "win";

// この手数までは時間延長なし
constexpr int _NO_EXTEND_TIME_PLY_LIMIT = 20;
#if defined(IS_DENRYUSEN_PLY_OFFSET)
constexpr int NO_EXTEND_TIME_PLY_LIMIT = _NO_EXTEND_TIME_PLY_LIMIT + DENRYUSEN_PLY_OFFSET;
#else
constexpr int NO_EXTEND_TIME_PLY_LIMIT = _NO_EXTEND_TIME_PLY_LIMIT;
#endif

// ==================================================
// global変数
// ==================================================

namespace {
	// HACK: g_tree に。
	std::unique_ptr<UctTree> global_tree;
	// HACK: ヘッダーにglobal変数置くのは極めて遺憾だが、ヘッダーに関数全部纏ってる方がやりやすくて...
	__Board g_board;
	// interruption関連
	std::atomic<uint64_t> g_playout_count;
	//std::atomic<bool> g_is_stop;

	std::atomic<bool> g_is_pondering;    // ponder 中である(現在の思考はponder である)。(なんかとりまatomic にしたけど、普通でええんじゃないか？)
	go_info_t g_go_info;
	stopwatch_t g_stop_watch;
	uint64_t g_last_pv_print_playout;
	int g_time_limit;        // ms
	int g_n_extend_time;     // 現在のroot局面において、何回探索延長をしたか

	int g_div_init;     // 初期値(最大値)
	int g_div_final;    // 最小値
	int g_extend_time_sf_ratio_x100;    // ratio of second to first. 次善手がこの割合より高い勝率なら探索延長。
	float g_extend_time_sf_ratio;    // ratio of second to first. 次善手がこの割合より高い勝率なら探索延長。

	//// ----------
	//// USI option
	//// ----------
	
	// 探索, 時間制御 等
	bool g_is_debug;
	bool g_usi_ponder;
	bool g_stochastic_ponder;
	int g_n_threads;
	int g_batch_size;
	int g_minimum_thinking_time;
	int g_minimum_thinking_time_ratio_x100;    // time_limit の内絶対に思考する時間の割合。
	float g_minimum_thinking_time_ratio;
	int g_network_delay;    // 通信遅延を考慮して、これだけ余裕を持つ。
	uint64_t g_const_playout;
	int g_print_pv_per_playout;

	// puct の探索パラメータ
	int g_puct_c_base;
	int g_puct_c_init_x100;
	float g_puct_c_init;
	int g_puct_c_fpu_reduction_x100;
	float g_puct_c_fpu_reduction;

	int g_puct_c_base_root;
	int g_puct_c_init_root_x100;
	float g_puct_c_init_root;
	int g_puct_c_fpu_reduction_root_x100;
	float g_puct_c_fpu_reduction_root;

	int g_softmax_temperature_x100;            // USI_option ではn 倍の値を登録。
	float g_softmax_temperature;    // 内部ではn で割った値を使用。
	int g_draw_value_black_x1000;
	float g_draw_value_black;
	int g_draw_value_white_x1000;
	float g_draw_value_white;
	//int g_init_value_x100;
	//float g_init_value;

	// model, book
	bool g_is_fp16;
	std::string g_onnx_model;
	bool g_own_book;
	std::string g_book_filename;
	int g_book_moves;    // 何手目まで定跡を用いるか
	int g_book_black_eval_limit;
	int g_book_white_eval_limit;
	int g_book_depth_limit;
	bool g_book_check_depth_limit_last;

	// dfpn
	int g_draw_ply;

	int g_dfpn_self_hash_size_mb;
	int g_dfpn_enem_hash_size_mb;
	int g_root_dfpn_n_threads;
	int g_root_dfpn_max_depth;
	uint64_t g_root_dfpn_max_node;

	int g_pv_dfpn_n_threads;    // 使用できる合計スレッド数
	int g_pv_dfpn_n_threads_per_node;
	int g_pv_dfpn_max_depth;
	uint64_t g_pv_dfpn_max_node;

	// mutex
	std::mutex g_pos_mutexes[POS_MUTEX_NUM];
}

// HACK: ちょっとこの上部のglobal 関数辺りリファクタリング(整理) したいよね。関数でごった返してる。





// ==================================================
// global 関数
// ==================================================

// プロトタイプ宣言
void print_ucb(uct_child_info_t* info, const UctNode* node);
UctNode* print_child_and_transition(UctNode* parent, const int child_idx);

// TODO: これってええんか...?
// 外部に探索木への参照を返す。
std::unique_ptr<UctTree>& get_global_tree() { return global_tree; }

// NOTE: thread safe (https://yohhoy.hatenablog.jp/entry/2013/12/15/204116)
// position のkey に応じたmutex を返す。
inline std::mutex& get_position_mutex(const Position* pos)
{
	// NOTE
	//     : pos->getKey() & (MUTEX_NUM - 1) が簡易的なHash になってる感じ？
	//       それによって、同じ局面を同時に操作しない、的な...?
	return g_pos_mutexes[pos->getKey() & (POS_MUTEX_NUM - 1)];
}

inline void set_g_is_pondering(const bool is_pondering [[carries_dependency]] ) { g_is_pondering = is_pondering; }
inline [[carries_dependency]] bool get_g_is_pondering() { return g_is_pondering; }

// TODO
//     : unique_ptr のスレッドセーフな操作
// NOTE
//     : 恐らく実装上thread safe なはず。
//         : 少なくともglobal 変数への書き込みは無い。
// root node は展開されているものとする。(一応未評価でも動くはずだが、意味のある指し手は返ってこないはず)
// @arg ptr_bestmove
//     : bestmove が欲しいときは、ここにbestmove を貰うポインタを置く。
// @arg ptr_pondermove
//     : pondermove が欲しいときは、ここにpondermove を貰うポインタを置く。
//     : pondermove が分からなければ、MoveNone がset される。
// @arg ptr_bestvalue
//     : bestvalue が欲しいときは、ここにbestvalue を貰うポインタを置く。
//     : root node が未展開なら、適当に0.5 がset される。これはあまり深く考えてない。(現状bestvalue は使ってないので。。。)
inline void get_bestmove_and_print_pv(Move* ptr_bestmove, Move* ptr_pondermove, float* ptr_bestvalue) {
	//// ------------------------------
	//// 1) PV 以外を先に求める。
	//// ------------------------------

	if (!global_tree->current_head->is_expanded()) {    // root が未展開なら何も出来ないので終了
		if (ptr_bestmove   != nullptr) { *ptr_bestmove   = Move::moveNone(); }
		if (ptr_pondermove != nullptr) { *ptr_pondermove = Move::moveNone(); }
		if (ptr_bestvalue  != nullptr) { *ptr_bestvalue  = 0.5;              }
		return;
	}

	// 訪問回数が最大の手を選択する。
	const int root_bestmove_idx = global_tree->current_head->get_bestmove_idx();
	if (root_bestmove_idx < 0) {
		sync_cout << "info string Error! root_bestmove_idx < 0" << std::flush << sync_endl;
	}

	const float root_bestvalue = global_tree->current_head->get_winrate(root_bestmove_idx);
	const Move root_bestmove = global_tree->current_head->child_infos[root_bestmove_idx].move;

	// ポインタにセット
	if (ptr_bestmove   != nullptr) { *ptr_bestmove   = root_bestmove;    }
	if (ptr_pondermove != nullptr) { *ptr_pondermove = Move::moveNone(); }    // 初期化
	if (ptr_bestvalue  != nullptr) { *ptr_bestvalue  = root_bestvalue;   }

	float cp;
	if (root_bestvalue == 1) {
		cp = 3e4;
	}
	else if (root_bestvalue == 0) {
		cp = -3e4;
	}
	else {
		// TODO: eval_coef
		cp = int(-logf(1.0 / root_bestvalue - 1.0) * 600);
	}

	const auto& crrt_playout_count = g_playout_count.load();
	const double& elapsed_time_ms = g_stop_watch.elapsed_ms_interim();
	const double& nps = (1000 * crrt_playout_count) / elapsed_time_ms;

	//// ------------------------------
	//// 2) PV を求める。
	//// ------------------------------

	std::string pv = root_bestmove.toUSI();
	int depth = 1;
	int crrt_bestmove_idx = root_bestmove_idx;
	UctNode* current_pv_node = global_tree->current_head;
	UctNode* next_pv_node = nullptr;    // current_pv_node での現状の最善手を指した時の、遷移先のnode
	while (true) {
		//// 1) 可能なら最善子ノードに遷移する

		if (!current_pv_node->child_infos[crrt_bestmove_idx].is_exist()) {
			break;
		}

		// 既にnode が存在しているなら、create_child(make_unique) とのデータ競合は起きえないのでlock とか要らん。
		next_pv_node = current_pv_node->child_nodes[crrt_bestmove_idx].get();
		if (next_pv_node->move_count == 0) {
			// 現在のnode こそ評価されているものの、次のnode には一度も訪問していないならそれはPV とは言えない。
			// (TODO:この時、現在のnode の訪問数==1のはず... 確認しよう。)
			break;
		}

		current_pv_node = next_pv_node;    // 次のnode へ遷移

		//// 2) 可能なら遷移先での最善手を取得

		// TODO
		//     : 展開されていて、且つ対応する子ノードが存在しているなら、nn にも評価されている気がするので、
		//       展開されているか否かだけ確認すれば良い気もする。。。(一旦念の為このままにしている。)
		if (!current_pv_node->is_expanded() || !current_pv_node->is_evaled_by_nn()) {    // 未展開 || 未評価
			break;
		}

		crrt_bestmove_idx = current_pv_node->get_bestmove_idx();
		if (crrt_bestmove_idx < 0) {
			sync_cout << "info string Error! crrt_bestmove_idx < 0 (in while{})" << std::flush << sync_endl;
		}

		const auto& crrt_bestmove = current_pv_node->child_infos[crrt_bestmove_idx].move;
		pv += " " + crrt_bestmove.toUSI();
		++depth;    // NOTE: pv に手が付け加えられるタイミングは、depth が増えるタイミングである。

		if (depth == 2 && ptr_pondermove != nullptr) { *ptr_pondermove = crrt_bestmove; }
	}

	//// ------------------------------
	//// 3) 出力
	//// ------------------------------
	
	// NOTE: やねうら王の順番
	//       <2:info depth 1 seldepth 1 score cp 1054 nodes 55 nps 27500 time 2 pv 4e5d
	sync_cout
		<< "info"
		<< " depth " << depth
		<< " score cp " << cp
		<< " nodes " << crrt_playout_count
		<< " nps " << positive_decimal_to_int_str(nps)
		<< " time " << positive_decimal_to_int_str(elapsed_time_ms)
		<< " pv " << pv << std::flush << sync_endl;
}

// ==================================================
// ParallelPuctSearcherGroup
// ==================================================

// NOTE
//     : https://learn.microsoft.com/ja-jp/cpp/error-messages/compiler-errors-1/compiler-error-c2027?view=msvc-170
//       クラスが相互に参照する場合、プロトタイプ宣言(前方宣言)があればメンバ変数にポインタを取ることは出来る。
//       然し、直接参照してメンバ関数を用いる、等といった場合は完全に定義されてからで無ければならない。
bool ParallelPuctSearcherGroup::_isready() {
	if (_nn != nullptr) {
		delete _nn;
	}

	_nn = new NNTensorRT(g_onnx_model, g_batch_size, g_is_fp16);

    // https://cpprefjp.github.io/reference/vector/vector/resize.html
	_searchers.resize(g_n_threads);

	// _shared_stop を子供たちに共有。
	for (auto&& searcher : _searchers) {
		searcher.set_shared_stop(&_shared_stop);;
	}

	bool searchers_isready = true;
	for (auto& searcher : _searchers) {
		searchers_isready &= searcher._isready(this);
	}
	return searchers_isready;
}

void ParallelPuctSearcherGroup::run() {
	_shared_stop.store(false);

	for (int i = 0; i < g_n_threads; ++i) {
		// NOTE
		//     : https://teratail.com/questions/272026
		//       "C2672 'invoke': 一致するオーバーロードされた関数が見つかりませんでした" なるエラーは、引数が与えられていない際に出たりするらしい。
		//       そして今回のラムダ式は引数にconst int& i を取るので、当然その引数を二つ目以降の引数で渡してやらんといかん。忘れてた。
		// HACK: これ、キャプチャリストでコピーして渡してやれば、引数i 要らんのでは？

		_searcher_ths.emplace_back(
			[this](const int& i) {
				this->_searchers[i].parallel_search(i == MONITORING_THREAD_ID);
			},
			i
		);
	}
}

void ParallelPuctSearcherGroup::join() {
	for (auto& th : _searcher_ths) {
		th.join();
	}
	_searcher_ths.clear();
}

bool ParallelPuctSearcherGroup::nn_forward(const int& batch_size, const feature_t input, float* output_p, float* output_v) {
	_gpu_mtx.lock();
	const auto&& status = _nn->forward(batch_size, input, output_p, output_v);
	_gpu_mtx.unlock();
	return status;
}

// YET: 以下の定義だけをここに。
// ==================================================
// ParallelPuctSearcher
// ==================================================
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

// NOTE
//     : unsafe
//         : この関数自体を並列で動かしてはならない。
//           (∵g_time_limit, g_n_extend_time への書き込みがある。)
// 探索を終了すべきか否か()
// @return
//     : 一応、探索終了なら1, 続行なら0 を返すが、これはあくまで途中で関数を抜けるための物で、
//       あんまりちゃんと検証してないのでこの値は使うべきでない。
//       -> それなら普通にvoid でreturn; したらええやんけ。何言うとんねん。
bool ParallelPuctSearcher::_should_exit_search() {
	//// debug
	//return false;

	// TODO
	//     : https://silight.hatenablog.jp/entry/2014/10/13/164915
	//       基本に立ち返ってね。一番の考え方はrelease, acquire 見たら分かりやすいよね。
	//       p1 でstore(release) したなら、release 以前の書き込み走査全てが、p1以外のスレッドからも一貫して見えてるんだよな。
	//       ほんでもって、p2 でload(aquire) したならそれ以降の読み込み操作が先走ってそのload(acquire) より先にはされないんだよな。
	//       だとしたらよ、以下はどうなんねん。
	//       int data = 0;
	//       void th1() {
	//           data = 9;
	//           flag.store(true, r)
	//       }
	//       void th2() {
	//           isOk = flag.load(acquire);
	//           if (isOk) {
	//               data;    // もしth2()だけなら、"この変数の値のload (読み込み)はif より前に事前に(例えば関数の先頭で)行っておいて、
	//                        // 必要なったらpre_fetch() した値を取り出す"、とかでも行けて、それを防ぐためのメモリフェンスってこと？
	//                        // だとしたら、x = data; が事前に成される可能性があるってこと？そんなことなくない？
	//                        // x がif のスコープ内の変数なら良いかも知らんが、例えばglobal scope のx とかに、
	//                        // 実行されるか分からないif 文の中身を先に実行するのってダメだよな？
	//           }
	//       }

	//// ------------------------------
	//// 1) 必要な定数の取得
	//// ------------------------------

	// NOTE
	//     : 探索木を再利用しているので、mvoe_count0 - mvoe_count1 が、
	//       見かけの思考時間に見合わない(思考時間に対して大きすぎる)場合がある。
	//       そういう時は、time_limit の半分も使わずに 探索が切り上げられる事がある。
	// 残りの探索で、次善手が最善手を超える可能性があるなら、探索は続行
	const auto& crrt_playout_count = g_playout_count.load(std::memory_order_relaxed);
	const auto& elapsed_time_ms = g_stop_watch.elapsed_ms_interim();
	const auto& estimated_extra_playout = crrt_playout_count * (g_time_limit - elapsed_time_ms) / elapsed_time_ms;

	const auto& crrt_head = global_tree->current_head;
	auto [first_idx, second_idx] = crrt_head->get_1th_2th_idx();

	//// ------------------------------
	//// 2) check win, lose
	//// ------------------------------

	if (first_idx < 0 && second_idx < 0) {    // どの指し手を指し手も負け
		// debug
		sync_cout << "info string _should_exit_search(): parent lose." << std::flush << sync_endl;

		// 親が負け確定
		return true;
	}
	else if (first_idx < 0) {    // first_idx < 0 && second_idx >= 0
		// DEBUG
		//     : 最善手は無いのに次善手のみ存在するという結果は違法。念の為チェック。
		//     : std::endl 自体flush するはずだが、念の為明示的にflush。まぁ正直意味ない希ガス。
		sync_cout << "info string Error! first_idx < 0 && second_idx >= 0" << std::flush << sync_endl;
	}

	const auto& first_child_info = crrt_head->child_infos[first_idx];
	if (first_child_info.is_lose()) {
		// debug
		sync_cout << "info string _should_exit_search(): parent win." << std::flush << sync_endl;

		// 親が勝ち確定
		return true;
	}

	//// ------------------------------
	//// 3) 制限無し(infinite, ponder), 定数的制限(プレイアウト制限, byoyomi, minimum_thinking_time)
	//// ------------------------------

	//if (g_go_info.is_infinite || get_g_is_pondering()) {    // 時間制限が(実質的に)無いので、止められるまで永遠に実行。
	//	return false;
	//}

	// 非ponder で、プレイアウト数制限が有効である
	if (g_const_playout > 0) {
		if (crrt_playout_count >= g_const_playout) {
			// debug
#if defined(VERBOSE)
			sync_cout << "info string _should_exit_search(): playout limit" << std::flush << sync_endl;
#endif

			return true;
		}
		return false;
	}

	// TDOO: 本当は、inf の時はstopコマンド より先にbestmove 返すべきじゃないらしいね。
	// 時間制限が無くても、勝敗が決してるなら探索を中断したいので、ここまで待ってから抜ける。
	if (g_go_info.is_infinite || get_g_is_pondering()) {    // 時間制限が(実質的に)無いので、止められるまで永遠に実行。
		return false;
	}

	// 非ponder で、持ち時間を使い切って、秒読み将棋に入った場合。
	if (g_go_info.get_time(g_board.turn()) == 0 && g_go_info.is_byoyomi()) {
		// HACK: 本当は、思考途中で持ち時間分を使い切ったら、その時点からギリギリまで秒読み時間を使い切るべきだけど、まぁ秒読みとかせんし。。。
		// 秒読みなら今の持ち時間が未来に持ち越されることは無いので、ギリギリまで時間を使い切る。
		if (elapsed_time_ms >= g_time_limit) {
#if defined(VERBOSE)
			sync_cout << "info string _should_exit_search(): byoyomi time limit" << std::flush << sync_endl;
#endif

			return true;
		}
		return false;
	}



	// NOTE
	//     : 時間に余裕があれば、最小思考時間以上は強制的に思考させるべきだが、
	//       制限時間を越えて、思考を強制させることは出来ない。
	//       (場合によってはnetwork_delay が聞かなかったり、
	//        5分+秒読み2秒で最小思考時間を3秒とかの場合に秒読みに入ったら(持ち時間5分使い切ったら)切れ負けてしまう)
	// 勝敗が決していないなら、最小思考時間以上 , g_time_limit の0.1倍 は思考する。
	if (elapsed_time_ms < my_min(g_minimum_thinking_time, g_time_limit * g_minimum_thinking_time_ratio) || elapsed_time_ms * 10 < g_time_limit) {
		return false;
	}

	// NOTE
	//     : 候補手が1 つで、その指し手で勝ち確定ではない時、ここまででreturn true; しない。
	//       だが、その場合second_idx = -1 なので、アクセス違反となる。
	//       今のところ、候補手が1つの場合は即座にその指し手を返すので、
	//       この関数が呼ばれることは無く、問題ない。

	//// ------------------------------
	//// 4) 探索延長するべきなら延長。さもなければ終了。
	//// ------------------------------

	// 合法手が1つだけ or 他の指し手が全部負け と分かったので指せる手が一つしかないので終了。
	if (second_idx < 0) {
		return true;
	}

	const auto& child_info0 = crrt_head->child_infos[first_idx];
	const auto& child_info1 = crrt_head->child_infos[second_idx];
	const auto& move_count0 = crrt_head->get_move_count(first_idx);
	const auto& move_count1 = crrt_head->get_move_count(second_idx);
	const auto& winrate0 = crrt_head->get_winrate(first_idx);
	const auto& winrate1 = crrt_head->get_winrate(second_idx);
	const auto& move0 = crrt_head->get_move(first_idx);
	const auto& move1 = crrt_head->get_move(second_idx);

	// 残り時間で、次善手が最善手を越す可能性がまだある。
	if (move_count0 - move_count1 <= estimated_extra_playout) {
		return false;
	}

	// HACK: 1.5, 0.1, 20 がマジックナンバーになってるのでconstexprして！
	// まだ次善手が逆転可能と思われる割合
	float comebackable_rate = 1.5;
	if (winrate0 < winrate1) {
		// 勝率が勝っているなら、逆転はよりしやすいはず。
		//comebackable_rate += 0.1;    // 一旦コメントアウト。計測してないのであるなしでどう変化するのか分からん。
	}

	// 次善手が逆転出来なさそうになっても、一度はチャンスを与える。
	// 探索延長したことが無い
	// && 21手目以降(NO_EXTEND_TIME_PLY_LIMIT 手までは延長しない)
	// && 2倍に探索延長するだけの時間の余裕がある
	// && (次善手の勝率が、最善手の勝率 * 1.0 より高い
	//     || 最善手のmove_count が、次善手の1.5倍よりは低い)
	if (g_n_extend_time == 0
		&& g_board.ply() > NO_EXTEND_TIME_PLY_LIMIT
		&& g_time_limit * 2 < g_go_info.get_time(g_board.turn())
		&& (winrate1 > winrate0 * g_extend_time_sf_ratio
		    || move_count0 < move_count1 * comebackable_rate)
	) {
		const int&& estimated_extra_playout_if_time_double = crrt_playout_count * (g_time_limit * 2 - elapsed_time_ms) / elapsed_time_ms;

		if (move_count0 - move_count1 <= estimated_extra_playout_if_time_double) {
			g_time_limit *= 2;    // ===== modify global ====
			g_n_extend_time += 1;    // ===== modify global ====
			sync_cout << "info string extend time, "
				<< "[move:" << __move_to_usi(move0.value()) << ", " << __move_to_usi(move1.value()) << "]"
				<< "[count:" << move_count0 << ", " << move_count1 << "]"
				<< "[winrate:" << winrate0 << ", " << winrate1 << "]"
				<< "[prob:" << child_info0.probability << ", " << child_info1.probability << "]"
				<< "[" << estimated_extra_playout << ", " << crrt_playout_count << "]"
				<< sync_endl;
			return false;
		}
	}

	sync_cout << "info string pruning, "
		<< "[move:" << __move_to_usi(move0.value()) << ", " << __move_to_usi(move1.value()) << "]"
		<< "[count:" << move_count0 << ", " << move_count1 << "]"
		<< "[winrate:" << winrate0 << ", " << winrate1 << "]"
		<< "[prob:" << child_info0.probability << ", " << child_info1.probability << "]"
		<< "[" << estimated_extra_playout << ", " << crrt_playout_count << "]"
		<< sync_endl;

	// 二度目のチャンスはない。探索を止める。
	// debug
#if defined(VERBOSE)
	sync_cout << "info string _should_exit_search(): normal stop" << std::flush << sync_endl;
#endif

	return true;
}

// TODO: player の_isready で呼ばれる
bool ParallelPuctSearcher::_isready(ParallelPuctSearcherGroup* parent_parallel_puct_searcher_group) {
	_parent_parallel_puct_searcher_group = parent_parallel_puct_searcher_group;
	if (_parent_parallel_puct_searcher_group == nullptr) {
		return false;
	}

	_cuda_free_input();
	cudaFree(_output_policy);
	cudaFree(_output_value);

#if defined (FEATURE_V2)
	// HACK: dlshogi みたいなtypedef をすれば、ここがsizeof() で統一出来たんやろうな。。。
	cudaHostAlloc((void**)&(std::get<0>(_input)), SIZEOF_FEATURE1 * g_batch_size, cudaHostAllocPortable);
	cudaHostAlloc((void**)&(std::get<1>(_input)), SIZEOF_FEATURE2 * g_batch_size, cudaHostAllocPortable);
#else
	cudaHostAlloc((void**)&_input, sizeof(float) * N_FEATURE_WHC * g_batch_size, cudaHostAllocPortable);
#endif

	cudaHostAlloc((void**)&_output_policy, sizeof(float) * N_LABEL_SIZE * g_batch_size, cudaHostAllocPortable);
	cudaHostAlloc((void**)&_output_value, sizeof(float) * 1 * g_batch_size, cudaHostAllocPortable);

	_eval_queue = std::vector<UctNodeAndFeatureAndColor>(g_batch_size);
	_current_queue_idx = 0;
	for (int i = 0; i < g_batch_size; ++i) {
		auto&& e = _eval_queue[i];
		e.first = nullptr;
#if defined (FEATURE_V2)
		// i番目のbatch に対応したメモリ領域を紐づける。
		std::get<0>(e.second.first) = std::get<0>(_input) + N_FEATURE1 * i;
		std::get<1>(e.second.first) = std::get<1>(_input) + N_FEATURE2 * i;
#else
		e.second.first = _input + N_FEATURE_WHC * i;    // i番目のbatch に対応したメモリ領域を紐づける。
#endif
	}

	return true;
}


// UCTSearcher::ParallelUctSearch()
// NOTE: 並列で呼ばれる
void ParallelPuctSearcher::parallel_search(const bool is_monitoring_thread) {
	// TODO: これ、_go() でやるべきな気もするな。
	//       -> いや、あれだ、queue_node(), eval_node() がsearcher 固有なので、Player は使えない。。。
	// 評価されていないなら評価する
	std::mutex& root_pos_mtx = get_position_mutex(&(g_board.pos));
	root_pos_mtx.lock();
	//std::cout << "info string is_evaled_by_nn = " << bts(global_tree->current_head->is_evaled_by_nn()) << std::endl;
	if (!global_tree->current_head->is_evaled_by_nn()) {
		queue_node(global_tree->current_head, g_board);
		eval_node();
	}
	root_pos_mtx.unlock();


	////// debug
	//if (global_tree->current_head->is_expanded() && global_tree->current_head->is_evaled()) {
	//	sync_cout << "info: [eval_node(root) done] print_ucb(head)" << sync_endl;
	//	print_ucb(nullptr, global_tree->current_head);

	//	sync_cout << "info: [eval_node(root) done] print_ucb(head->debug_tag_idx_1)" << sync_endl;
	//	const int debug_tag_idx_1 = 71;
	//	UctNode* next_node_1 = print_child_and_transition(global_tree->current_head, debug_tag_idx_1);

	//	if (next_node_1 != nullptr) {
	//		sync_cout << "info: [eval_node(root) done] print_ucb(head->debug_tag_idx_1->debug_tag_idx_2)" << sync_endl;
	//		const int debug_tag_idx_2 = 84;
	//		UctNode* next_node_2 = print_child_and_transition(next_node_1, debug_tag_idx_2);
	//	}
 //   }

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
		// debug
		// ログで差分取るためにdlshogi に出力合わせる。
#ifdef DEBGU_PUCT_20231125_2
		std::cout << "info po_info.count=" << g_playout_count << std::endl;
		uint64_t prevplyout = g_playout_count.load();
#endif

		trajectories_batch.clear();
		trajectories_batch_discarded.clear();
		current_batch_top = -1;    // top の要素のidx
		//std::vector<__Board> board_cpys(_batch_size, _board);
		//std::cout << "info string loop start" << std::endl;
		for (int i = 0; i < g_batch_size; ++i) {
			//std::cout << "info string loop[" << i << "]" << std::endl;

			// TODO: これ、本当はboard は参照渡しして、uct_search() でundoMove() (というより、pop())もするようにすべき。
			// 明示的なコピーコンストラクタの呼出し
#ifdef DEBUG_
			std::cout << "info: start copy constructer" << std::endl;
#endif
			//_check_state_info(_board, "search() (before cpy)");	
			// NOTE: やっぱこれ、local だったのが良くなかったんやで。
			// HACK: デフォルトコンストラクタってある？あんま良く分かってない
			// TODO: g_board への書き込み操作が被らないか確認
			__Board board_cpy(g_board);
			//__Board& board_cpy = board_cpys[i];
			// _check_state_info(board_cpy, "search() (after cpy)");

			// TODO: これで本当にからのtrajectory 入ってるかは怪しい。
			trajectories_batch.emplace_back(std::initializer_list<UctNodeAndMove>{});
			++current_batch_top;

#ifdef DEBGU_PUCT_20231125_2
			std::cout << "[" << g_playout_count << "] ";
#endif
			float result = uct_search(board_cpy, nullptr, global_tree->current_head, trajectories_batch[current_batch_top]);

#ifdef DEBGU_PUCT_20231125_2
			std::cout << std::endl;
#endif

			if (result == RESULT_DISCARDED) {
				trajectories_batch_discarded.emplace_back(trajectories_batch[current_batch_top]);
			}
			else {
				++g_playout_count;
			}
			if (result != RESULT_QUEUING) {    // 推論待ち出なければ、既にbackup済みなので削除。
				trajectories_batch.pop_back();
				--current_batch_top;
			}
			//if (trajectories_batch_discarded.size() > g_batch_size / 2) {    // batch_size の半分以上を破棄が占める場合、とっとと評価フェーズに移行。
			//	break;
			//}
		}
		//std::cout << "info string loop done" << std::endl;
#ifdef DEBUG_
		std::cout << "playout_count = " << playout_count << std::endl;
		//std::cout << "info debug: for all done" << std::endl;
#endif
#ifdef DEBUG_
		std::cout << "info string debug: eval_node() start" << std::endl;
#endif

		// 評価
		if (trajectories_batch.size() > 0) {
			eval_node();
		}

		////// debug
		//if (global_tree->current_head->is_expanded() && global_tree->current_head->is_evaled()) {
		//	sync_cout << "info: [eval_node() done] print_ucb(head)" << sync_endl;
		//	print_ucb(nullptr, global_tree->current_head);

		//	sync_cout << "info: [eval_node() done] print_ucb(head->debug_tag_idx_1)" << sync_endl;
		//	const int debug_tag_idx_1 = 71;
		//	UctNode* next_node_1 = print_child_and_transition(global_tree->current_head, debug_tag_idx_1);

		//	if (next_node_1 != nullptr) {
		//		sync_cout << "info: [eval_node() done] print_ucb(head->debug_tag_idx_1->debug_tag_idx_2)" << sync_endl;
		//		const int debug_tag_idx_2 = 84;
		//		UctNode* next_node_2 = print_child_and_transition(next_node_1, debug_tag_idx_2);
		//	}
		//}

#ifdef DEBUG_
		std::cout << "info string debug: eval_node() done" << std::endl;
#endif
		//std::cout << "info string cancel VIRTUAL_LOSS start" << std::endl;
		// discarded された奴のVirtual Loss を元に戻す 
		// (破棄された奴は、無かったことにしたい。)
		for (const auto& trajectory : trajectories_batch_discarded) {
			for (const auto& node_and_move : trajectory) {
				node_and_move.first->move_count -= VIRTUAL_LOSS;    // TODO20231108: node のmove_count
				atomic_fetch_sub(&node_and_move.first->child_infos[node_and_move.second].move_count, VIRTUAL_LOSS);
			}
		}
		//std::cout << "info string cancel VIRTUAL_LOSS done" << std::endl;

		// NOTE
		//     : 書き込みはupdate_result() のみであり、
		//     : queueing だった各trajectory について、探索結果を反映していく。
		// queueing した奴のbackup(VirtualLoss を元に戻して、平均報酬を更新)
		//std::cout << "info string backup start" << std::endl;
		float result;
		for (const auto& trj : trajectories_batch) {
			result = -1;    // TOOD: RESULT_NONE = -1 として他の値と衝突しないか。
			//       (まぁ、この初期値は一切使ってないから、そもそも初期化せんでええ気もする。)
			for (auto it = trj.rbegin(), it_rend = trj.rend(); it != it_rend; ++it) {
				auto node = it->first;
				auto next_index = it->second;
				if (it == trj.rbegin()) {
					// NOTE
					//     : このバックアップを行うのはresult = QUEUEING の場合のみで、
					//        value = VALUE_WIN 等の[0, 1] の範囲外の値が代入されている場合は行われない(∵result != QUEUEING)。
					// このvalue は、今回の推論によるものである。
					// queue_node(child_node, board); としてることからも分かるように、
					// 今回の推論によるものは子ノードの手番からのvalue なので反転する。
					result = 1 - node->child_nodes[next_index]->value;

					// debug
#ifdef DEBGU_PUCT_20231125_0
					if ((g_playout_count >= 2024)) {
						std::cout
							<< "[" << prevplyout << "]"
							<< " value=" << node->child_nodes[next_index]->value
							<< std::endl;
						std::cout << traj_to_str(trj) << std::endl;
					}
#endif
#ifdef DEBGU_PUCT_20231125_1
					if (trj[0].first->child_infos[trj[0].second].move.toUSI() == "2b3b") {
						//std::cout
						//	<< "[" << prevplyout << "]"
						//	<< " value=" << node->child_nodes[next_index]->value
						//	<< std::endl;
						std::cout << "[" << prevplyout << "] " << traj_to_str(trj) << std::endl;
					}
#endif
				}
				update_result(node, next_index, result);
				result = 1 - result;    // NOTE: 手番に応じて反転？(@20230819)
			}

			// debug
#ifdef DEBGU_PUCT_20231125_2
			++prevplyout;
#endif
		}
		//std::cout << "info string backup done" << std::endl;

		//std::cout << "info string should_exit_search() start" << std::endl;
		if (is_monitoring_thread) {
			if (_should_exit_search()) {
				// _shared_stop により全てのスレッドを停止(当然、自分のすぐ次でbreak;)。
				this->set_shared_stop_true();
			}
		}
		//std::cout << "info string should_exit_search() done" << std::endl;

		if (_shared_stop->load()) {
			break;
		}

		// pv表示
		//std::cout << "info string get_bestmove_and_print_pv() start" << std::endl;
		if (is_monitoring_thread) {
			if (g_print_pv_per_playout > 0) {
				// TODO: ここ、フェンス要るんかね？(そりゃ、厳密にせんでも良さげやからそういう意味では要らんねんけど、厳密にする場合どうなん？要らん気がするけど。。)
				if (g_playout_count.load(std::memory_order_acquire) - g_last_pv_print_playout >= g_print_pv_per_playout) {
					g_last_pv_print_playout = g_playout_count;
					get_bestmove_and_print_pv(nullptr, nullptr, nullptr);
				}
			}
		}
		//std::cout << "info string get_bestmove_and_print_pv() done" << std::endl;

	}
}

// TODO
//     : board はpush してpop して再利用した方が速いはず。。。
// NOTE: 並列で呼ばれる
// NOTE
//     : result は、current_node の手番からみた結果である。
float ParallelPuctSearcher::uct_search(
	__Board& board, uct_child_info_t* current_info, UctNode* current_node, Trajectory& trajectory
) {
	// NOTE
	//     : 以下のresult は、当然だが、current_node の手番側から見た結果である。
	//       0 なら負け、1 なら勝ち。
	//     : 子ノードの実体がない 
	//       <=> current_node->child_nodes[next_index]->value は無い
	//       <=> 子ノードは未評価である
	//       -> 勝敗が明確でない限り、評価する。
	// TODO
	//     : 本当はVALUE_NONE とかで初期化すべきだけど、uct_node の奴を使うのなんかきもくねぇか。
	//    -> どっかに纏めたい。constant.hpp/.cpp として。
	//    -> てか、初期化する意味なくね？python だと定義まで必要だけど、c++ だから宣言だけでええやん？
	float result;

	std::mutex& pos_mtx = get_position_mutex(&(board.pos));

	pos_mtx.lock();
	// HACK: 評価する前にexpand するので、ここにきてまだ未展開の奴なんている？
	if (!current_node->is_expanded()) {    // 未展開
		current_node->expand_node(board);
	}

#if defined(DEBGU_PUCT_20231125_2) || false
	auto next_index = select_max_ucb_child(current_info, current_node);
#else
	const auto next_index = select_max_ucb_child(current_info, current_node);
#endif
	if (next_index < 0) {    // このノードは負け確定
#if defined(DEBGU_PUCT_20231125_2) || false
		next_index = 0;
#else
		// NOTE: (子ノードは訪問しないので加算しない。
		//       一応、一番マシな指し手を選んでそいつを指せば、こういう特別な処理は挟まんで良くなるけど、
		//       負け確の時にわざわざマシな手を計算するはコストが増える。
		//       かといって適当にnext_index==0 を返すと、適当な指し手の訪問回数が1回増えちゃうので地味にキモイ。
		//       (まぁ正直1回 増えるぐらい気にしなくて良いと思うけどネ))
		//     : 何なら、訪問しなかったことにして、
		//           result = 0;
		//           pos_mtx.unlock();
		//           return 1 - result;
		//       だけでも良い気もするな。
		// Virtual Loss を加算
		current_node->move_count += VIRTUAL_LOSS;

		result = 0;
	    // NOTE: queueing ではない場合、backup(探索木への探索結果の反映) はここで行う。
	    //       (backup は、queueing の場合はeval_node()後に行い、
	    //       discarded の場合はそもそもbackup は無理だしする必要もなく、それ以外の勝敗が決している場合はここで行う。)
	    update_result(current_node, next_index, result);

		// debug

#ifdef DEBGU_PUCT_20231125_0
		if ((g_playout_count >= 2024)) {
			std::cout
				<< "[" << g_playout_count << "]"
				<< " trajectory=" << trajectory.size()
				<< ", result=" << result
				<< ", AbsLose"
				<< std::endl;
			std::cout << traj_to_str(trajectory) << std::endl;
		}
#endif

		pos_mtx.unlock();
		return 1 - result;
#endif
}

	auto& next_child_info = current_node->child_infos[next_index];
	board.push(next_child_info.move);

	// debug
#ifdef DEBGU_PUCT_20231125_0
	//if (trajectory.size() == 0) {
	//	std::cout
	//		<< "[" << g_playout_count << "]"
	//		<< ", move=" << next_child_info.move.toUSI()
	//		<< ", move=" << next_child_info.move.value()
	//		<< std::endl;
	//}
#endif

	// Virtual Loss を加算
	current_node->move_count += VIRTUAL_LOSS;    // TODO20231108: node のmove_count
	atomic_fetch_add(&current_node->child_infos[next_index].move_count, VIRTUAL_LOSS);

	// 探索経路を記録
	trajectory.emplace_back(current_node, next_index);

	if (!current_node->child_nodes[next_index]) {    // child_nodes[next_index]の実体がない -> 実体を作成して評価
		UctNode* child_node = current_node->create_child_node(next_index);

		// NOTE: 全く同じ子node がここには入れる条件は、子ノードcurrent_node->child_nodes[next_index] の実体が無いこと。
		//       create_child_node() で実体を作ったので、次全く同じ子node がpos_mtx を獲得した時はここに入れず破棄される。
		//       -> つまり、これ以降のchild_node に対する如何なる操作もthread safe である。
		//          (∵実体があった場合(else) では破棄される限りnode に対する変更操作は一切なされない & 
		//          他に動いている可能性のある関数parallel_search() で全く同じ子ノードへの書き込みが行われるのはeval_node() 後ゆえ。)
		// TODO
		//     : dfpn を並列で回して適宜value の書き換えを行う場合、
		//       create_child_node() されてから、eval_node() されるまでの間のUctNode がPVに入ってくることがあるのかを確認すべし。
		//       if (PV に入ってくるのでクリティカルセクション) {
		//           value をatomic に ||
		//           value にアクセスするときにposition_mutex(これは微妙だろ) ||
		//           move の上位8 bit に埋め込む(この場合、move の読み取りと被らないか確認しないといけない) ||
		//           全く別の3bit 以上の変数を別で用意する(無難) ||
		//           その他(考えちう....)
		//       }
		//       else {
		//           value の値をwin, lose で書き換え
		//       }
		pos_mtx.unlock();

		// TODO: ここで最大手数引き分けもcheck
		// child_node の局面において引き分けをチェック。
		auto repetition_type = board.isDraw(16);

		// 最大手数引き分け。g_draw_ply+1 手目以降は指せない。
		if (board.ply() > g_draw_ply) {
			repetition_type = RepetitionDraw;
		}

		if (repetition_type == NotRepetition) {
			//const std::string sfen = board.toSFEN();
			if (board.is_nyugyoku() || board.mateMove(5)) {
				//child_node->value = VALUE_WIN;
				next_child_info.set_win();
				result = 0;

				// debug
#ifdef DEBGU_PUCT_20231125_0
				if ((g_playout_count >= 2024)) {
					std::cout
						<< "[" << g_playout_count << "]"
						<< " trajectory=" << trajectory.size()
						<< ", result=" << result
						<< ", NyugyokuOrMate5"
						<< std::endl;
					std::cout << traj_to_str(trajectory) << std::endl;
				}
#endif
#ifdef DEBGU_PUCT_20231125_2
				std::cout << "[NyugyokuOrMate5]";
				//if (board.is_nyugyoku()) {
				//	std::cout << "[Nyugyoku]";
				//}
				//else {
				//	std::cout << "[Mate5]";
				//}
#endif
			}
			else {
				child_node->expand_node(board);
				// TODO
				//     : 詰将棋エンジン化するなら、以下はlegal_moves_size にしなければならないはず。
				if (board.is_game_over()) {
					//child_node->value = VALUE_LOSE;
					next_child_info.set_lose();
					result = 1;


					// debug
#ifdef DEBGU_PUCT_20231125_0
					if ((g_playout_count >= 2024)) {
						std::cout
							<< "[" << g_playout_count << "]"
							<< " trajectory=" << trajectory.size()
							<< ", result=" << result
							<< ", gameover"
							<< std::endl;
						std::cout << traj_to_str(trajectory) << std::endl;
					}
#endif
#ifdef DEBGU_PUCT_20231125_2
					std::cout << "[Gameover]";
#endif
				}
				else {
					//std::cout << "info: queue_node()" << std::endl;
					// _check_state_info(board, "uct_search() (before queue_node())");
					queue_node(child_node, board);

					// debug
#ifdef DEBGU_PUCT_20231125_2
					//if (traj_to_str(trajectory) == "P*3d S*5g") {
					//	std::cout << "[P*3d S*5g][" << board.toSFEN() << "]";
					//}
					std::cout << "[Queueing]";
#endif

					// _check_state_info(board, "uct_search() (after queue_node())");
					//std::cout << "info debug: uct_search() return RESULT_QUEUING;" << std::endl;
					return RESULT_QUEUING;
				}
			}
		}
		else {    // 引き分け
			// FIXME
			//     : 引き分け局面では推論せずに、set_draw() したらそれでset_evaled() としてる。
			//       なので、例えば実際に引き分け局面がroot となると、
			//       select_max_ucb_child が出来ない(推論結果が無いのでucb が計算できない)
			//       -> 一番無難な対策は、引き分けは、新しい局面ではset_draw() して、推論。(return queueing)
			//     : lose の局面なら賞味何指し手も負けなので、何指し手も一旦は良いと思う。
			//     : win の局面はchild がis_lose の指し手を選べばよいので、ucb は計算できなくて良い。

			//std::cout << "[" << playout_count << "] result = Repetition" << std::endl;
			switch (repetition_type) {
			case RepetitionDraw:
				//child_node->value = VALUE_DRAW;
				next_child_info.set_draw();
				if (board.turn() == Black) {
					// 今先手番なら親は後手番なので、後手番での評価値を使う
					result = g_draw_value_white;
				}
				else {
					result = g_draw_value_black;
				}
				break;
			case RepetitionWin: case RepetitionSuperior:
				// TODO
				//     : 詰将棋エンジン化するなら、以下は勝ちとすべきでは無い気がする。
				//child_node->value = VALUE_WIN;
				next_child_info.set_win();
				result = 0;
				break;
			case RepetitionLose: case RepetitionInferior:
				//child_node->value = VALUE_LOSE;
				next_child_info.set_lose();
				result = 1;
				break;
			default:
				throw std::runtime_error("Error");
			}

			// debug
#ifdef DEBGU_PUCT_20231125_0
			if ((g_playout_count >= 2024)) {
				std::cout
					<< "[" << g_playout_count << "]"
					<< " trajectory=" << trajectory.size()
					<< ", result=" << result
					<< ", Rep"
					<< std::endl;
				std::cout << traj_to_str(trajectory) << std::endl;
			}
#endif
#ifdef DEBGU_PUCT_20231125_2
			std::cout << "[Rep]";
#endif
		}
		// queue_node() 以外であれば、勝敗が決しているので評価済みに。
		child_node->set_evaled();
	}
	else {    // child_nodes[next_index]の実体がある。 -> 評価されている
		// これ以降、非atomic な書き込みが無い (& 読み込む変数(子ノード)への書き込みがどこかで並列実行されていることはない。)
		pos_mtx.unlock();

		UctNode* next_node = current_node->child_nodes[next_index].get();
		assert(next_node != nullptr);
		// NOTE
		//     : 他のスレッドがこのnext_node をeval_node() する瞬間と被るとデータ競合だよな。
		if (!next_node->is_evaled()) {    // 実体があるのに評価が無い -> 評価待ち(RESULT_QUEUING)
			//std::cout << "info debug: uct_search() return RESULT_DISCARDED;" << std::endl;


			// debug
#ifdef DEBGU_PUCT_20231125_0
			if ((g_playout_count >= 2024)) {
				std::cout
					<< "[" << g_playout_count << "]"
					<< " trajectory=" << trajectory.size()
					<< ", Discarded"
					<< std::endl;
				std::cout << traj_to_str(trajectory) << std::endl;
			}
#endif

			// debug
#ifdef DEBGU_PUCT_20231125_2
			std::cout << "[Discarded]";
#endif

			return RESULT_DISCARDED;
		}

		if (next_child_info.is_win() /* next_node->value == VALUE_WIN */ ) {
			result = 0;

			// debug
#ifdef DEBGU_PUCT_20231125_0
			if ((g_playout_count >= 2024)) {
				std::cout
					<< "[" << g_playout_count << "]"
					<< " trajectory=" << trajectory.size()
					<< ", result=" << result
					<< ", AlreadyEvaled"
					<< std::endl;
				std::cout << traj_to_str(trajectory) << std::endl;
			}
#endif
#ifdef DEBGU_PUCT_20231125_2
			std::cout << "[AlreadyEvaled:Win]";
#endif
		}
		else if (next_child_info.is_lose() /* next_node->value == VALUE_LOSE */) {
			result = 1;

			// debug
#ifdef DEBGU_PUCT_20231125_0
			if ((g_playout_count >= 2024)) {
				std::cout
					<< "[" << g_playout_count << "]"
					<< " trajectory=" << trajectory.size()
					<< ", result=" << result
					<< ", AlreadyEvaled"
					<< std::endl;
				std::cout << traj_to_str(trajectory) << std::endl;
			}
#endif
#ifdef DEBGU_PUCT_20231125_2
			std::cout << "[AlreadyEvaled:Lose]";
#endif
		}
		else if (next_child_info.is_draw() /* next_node->value == VALUE_DRAW */) {
			if (board.turn() == Black) {
				// 今先手番なら親は後手番なので、後手番での評価値を使う
				result = g_draw_value_white;
			}
			else {
				result = g_draw_value_black;
			}

			// debug
#ifdef DEBGU_PUCT_20231125_0
			if ((g_playout_count >= 2024)) {
				std::cout
					<< "[" << g_playout_count << "]"
					<< " trajectory=" << trajectory.size()
					<< ", result=" << result
					<< ", AlreadyEvaled"
					<< std::endl;
				std::cout << traj_to_str(trajectory) << std::endl;
			}
#endif
#ifdef DEBGU_PUCT_20231125_2
			std::cout << "[AlreadyEvaled:Draw]";
#endif
		}
		else {
			result = this->uct_search(board, &next_child_info, next_node, trajectory);
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

	// RESULT_QUEUING or RESULT_DISCARDED でないなら、この関数内でbackup する。

	// NOTE
	//     : queueing ではない場合、backup(探索木への探索結果の反映) はここで行う。
	//       (backup は、queueing の場合はeval_node()後に行い、
	//       discarded の場合はそもそもbackup は無理だしする必要もなく、それ以外の勝敗が決している場合はここで行う。)
	update_result(current_node, next_index, result);

	return 1 - result;
}

// なんかマクロの所為か、定義が見つかりませんってでるから、一旦これで警告なくしておく。上の奴が本物。
//float ParallelPuctSearcher::uct_search(__Board& board, UctNode* current_node, Trajectory& trajectory)
//{
//
//	return 0.0f;
//}

// These stunts are performed by trained professionals, do not try this at home.

// Fast approximate log2(x). Does no range checking.
// The approximation used here is log2(2^N*(1+f)) ~ N+f*(1+k-k*f) where N is the
// exponent and f the fraction (mantissa), f>=0. The constant k is used to tune
// the approximation accuracy. In the final version some constants were slightly
// modified for better accuracy with 32 bit floating point math.
inline float FastLog2(const float a) {
	unsigned int tmp;
	std::memcpy(&tmp, &a, sizeof(float));
	unsigned int expb = tmp >> 23;
	tmp = (tmp & 0x7fffff) | (0x7f << 23);
	float out;
	std::memcpy(&out, &tmp, sizeof(float));
	out -= 1.0f;
	// Minimize max absolute error.
	return out * (1.3465552f - 0.34655523f * out) - 127 + expb;
}

// Fast approximate ln(x). Does no range checking.
inline float FastLog(const float a) {
	return 0.6931471805599453f * FastLog2(a);
}


// NOTE
//     : thread safe
//          : 同じnode に対して並列で呼んでもthread safeなはず。
//			  もっと正確に言えば、legal_moves_size, node->child_infos[i].probability　への書き込みが並列で動いていない限りthread safeである。
//              : 前者について
//                -> つまり、同じnodeに対して、別のスレッドのexpand_node() とselect_max_ucb_child() が並列で動かなければok。
//                -> つまり、expand_node()とselect_max_ucb_child() が同じpos_mtx の元で行われれば、上記のような事態は起きない。
//                   (後、そもそも、展開されたnode でないとこの関数は適用できないはず。。。)
//              : 後者について
//                -> つまり、同じnode に対して、別のスレッドのeval_node() とselect_max_ucb_child() が配列で動かなればok。
//                -> これは、任意のnode がqueue_node() される回数が一回(即ちeval_node() も一回)であれば、今select_max_ucb_child() を実行しようとしているスレッド以外で、
//                   node がeval_node() される可能性は無い(∵select_max_ucb_child() はeval_node() された後でない限り実行されない)。
//     : #ifndef PYTHON_DLSHOGI2 の場合にのみ対応。
//       #ifdef PYTHON_DLSHOGI2 の場合には未対応。
//         : python-dlshogi2 は、微妙にucb の評価式がdlshogi と異なる。
//           かくなる私の実装もまた微妙に違うんだけどね。
//     : 並列で呼ばれる
// 
// 一番探索すべき指し手のindex を返す。
// @arg info, node
//     : 対応する局面は同じ。(同じ局面のinfo とnode)
//     : root の時はinfo==nullptr。(必要十分条件)
// @return
//     : 0未満(-1) なら、この局面は負け確定である。
//       そうでないなら、次に探索すべき指し手のindex
//       (こういうindex をマイナスで返す仕様、マジで危険なんだよな。。。)

// HACK: ひでぇ
static int n_logf_is_greater_than_approx = 0;    // logf が近似(FastLog) よりも大きかった回数
static float logf_is_greater_than_approx_loged_av = 0;    // そんなときのlog の中身   // TODO: インクリメンタルな方法。
static int n_approx_is_greater_than_logf = 0;
static float approx_is_greater_than_logf_loged_av = 0;
static int n_select_max_ucb_child_called = 0;

int select_max_ucb_child(uct_child_info_t* info, UctNode* node) {
#ifndef PYTHON_DLSHOGI2
	const int& total = node->legal_moves_size;

	float reward_mean, bonus;
	float current_max_ucb_score = -1;    // スコアが負にならない前提。
	int current_max_idx = -1;

	// atomic からload
	const auto node_move_count = node->move_count.load();
	const auto node_sum_value = node->sum_value.load();

	const auto this_puct_c_base = (info == nullptr ? g_puct_c_base_root : g_puct_c_base);
	const auto this_puct_c_init = (info == nullptr ? g_puct_c_init_root : g_puct_c_init);
	const float sqrt_sum = sqrtf(node_move_count);
#if defined(DEBGU_PUCT_20231125_2) || false
	const float C_s = FastLog((1.0 + node_move_count + this_puct_c_base) / this_puct_c_base) + this_puct_c_init;
#elif defined(DEBGU_PUCT_20231125_3)
	const float C_s = FastLog((1.0 + node_move_count + this_puct_c_base) / this_puct_c_base) + this_puct_c_init;
	const float C_s_my = logf((1.0 + node_move_count + this_puct_c_base) / this_puct_c_base) + this_puct_c_init;

	++n_select_max_ucb_child_called;
	if (C_s_my > C_s) {
		++n_logf_is_greater_than_approx;
		logf_is_greater_than_approx_loged_av += (
			((1.0 + node_move_count + this_puct_c_base) / this_puct_c_base) - logf_is_greater_than_approx_loged_av
			) / n_logf_is_greater_than_approx;
	}
	else if (C_s > C_s_my) {
		++n_approx_is_greater_than_logf;
		approx_is_greater_than_logf_loged_av += (
			((1.0 + node_move_count + this_puct_c_base) / this_puct_c_base) - approx_is_greater_than_logf_loged_av
			) / n_approx_is_greater_than_logf;
	}


	//std::cout << C_s << "/" << C_s_my << "/" << (1.0 + node_move_count + this_puct_c_base) / this_puct_c_base << std::endl;
#else
	// NOTE
	//     : FastLog に合わせて探索パラメータが調整されているため、dlshogi にあったFastLog を使う。
	//     : 9割以上の局面でFastLog の方が真の値より小さく、中身の平均は1.16553 (つまりlog(1.16553)) だった。
	//       残りの1割程度の局面ではFastLog の方が真の値より大きく、中身の平均は2.00802 だった。
	//const float C_s = logf((1.0 + node_move_count + this_puct_c_base) / this_puct_c_base) + this_puct_c_init;
	const float C_s = FastLog((1.0 + node_move_count + this_puct_c_base) / this_puct_c_base) + this_puct_c_init;
#endif
	
	// 未訪問node での値
	//const float init_reward_mean = (node->move_count == 0 ? 0 : node_sum_value / node_move_count);
	const float fpu_reduction = (
		info == nullptr ? g_puct_c_fpu_reduction_root : g_puct_c_fpu_reduction
		) * sqrtf(node->sum_visited_probability);
	const float init_reward_mean = (node_sum_value > 0 ?
		std::max(0.0f, (float)(node_sum_value / node_move_count) - fpu_reduction) : 0.0f);
	const float init_bonus = (node_move_count == 0 ? 1.0 : sqrt_sum);

	int n_child_win_moves = 0;    // 子が勝つ指し手の数(親が負ける指し手の数)
	for (int i = 0; i < total; ++i) {
		const auto& crrt_child_info = node->child_infos[i];
		if (crrt_child_info.is_win()) {    // 親の負け
			++n_child_win_moves;
			continue;
		}
		else if (crrt_child_info.is_lose()) {    // 親の勝ち
			if (info != nullptr) {
				info->set_win();    // OrNode
			}
#ifdef DEBGU_PUCT_20231125_2
			std::cout << "WIN";
#endif
			return i;
		}

		const auto node_child_move_count = crrt_child_info.move_count.load();
		const auto node_child_sum_value = crrt_child_info.sum_value.load();

		if (node_child_move_count == 0) {    // NOTE: この時、この子ノードは未評価のはず
			// TODO: 親ノードの平均報酬で近似する。
			//       -> https://tadaoyamaoka.hatenablog.com/entry/2020/01/13/121234
			//          dlshogi では0 より0.5 の方が強かったようで、試す価値はあるかもしれん。
	        //       -> いや、あれだな、向こうはpolicy の精度が良いからそうなるんじゃないかね。知らんけど。
			reward_mean = init_reward_mean;
			//reward_mean = g_init_value;    // 弱くなった。
			bonus = init_bonus;
		}
		else {
#if defined(CONSIDER_IS_DRAW)
			if (crrt_child_info.is_draw()) {
				reward_mean = 0.5;
			}
			else {
				reward_mean = node_child_sum_value / node_child_move_count;
			}
#else
			reward_mean = node_child_sum_value / node_child_move_count;
#endif

			bonus = sqrt_sum / (1 + node_child_move_count);
		}

		// dlshogi に桁落ちの仕方を揃える為に掛け算の順序変更。
		const float ucb_score = reward_mean + C_s * bonus * node->child_infos[i].probability;

		if (ucb_score > current_max_ucb_score) {
			current_max_ucb_score = ucb_score;
			current_max_idx = i;
		}
	}

	if (n_child_win_moves == total) {
		if (info != nullptr) {
			info->set_lose();    // AndNode
		}
#ifdef DEBGU_PUCT_20231125_2
		std::cout << "LOSE";
#endif
	}
	else {
#ifdef DEBGU_PUCT_20231125_2
		std::cout
			<< node->child_infos[current_max_idx].move.toUSI()
			<< "(" << current_max_ucb_score
			<< "," << node->child_infos[current_max_idx].sum_value
			<< "," << node->child_infos[current_max_idx].move_count
			//<< "," << C_s
			//<< "," << sqrt_sum
			<< "," << node->child_infos[current_max_idx].probability
			<< ") ";
#endif

		// fpu reduction
		atomic_fetch_add(&node->sum_visited_probability, node->child_infos[current_max_idx].probability);
	}

	return current_max_idx;

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

// NOTE
//     : 完全にthread safe
//     : VirtualLoss を戻して、平均報酬を更新
//     : 子ノードについては、next_idx が有効な数字(0以上)の時のみ更新を実行。
// 訪問回数、平均報酬の更新
void ParallelPuctSearcher::update_result(UctNode* node, int next_index, float result) const {
	// NOTE
	//     : https://en.cppreference.com/w/cpp/atomic/atomic/operator_arith2
	//       ページの先頭に小さく以下のように記されている。
	//       "member only of atomic<Integral﻿> specializations and [atomic<Floating> specializations(since C++20)]"
	//       c++20 以降ではfloat にも対応しているが、それまではinteger 専用の模様。
	atomic_fetch_add(&node->sum_value, result);
	if constexpr (1 - VIRTUAL_LOSS != 0) {
		node->move_count += 1 - VIRTUAL_LOSS;
	}

	if (next_index >= 0) {
		atomic_fetch_add(&node->child_infos[next_index].sum_value, result);
		if constexpr (1 - VIRTUAL_LOSS != 0) {
			node->child_infos[next_index].move_count += 1 - VIRTUAL_LOSS;
		}
	}
}

// NOTE
//     : 実装上thread safe (実装上あるのは後者のみで、それはthread safe)
//         : 同じオブジェクトから並列で呼ぶ場合、unsafe
//         : 同じオブジェクトからは一つだけ、異なるオブジェクトから並列で呼ぶ場合、thread safe。
// @arg node: 未評価のnode
// @arg board: node に対応した局面
void ParallelPuctSearcher::queue_node(UctNode* node, __Board& board) {
	// _check_state_info(board, "queue_node() (ref arg)");
	// NOTE
	//     : https://en.cppreference.com/w/cpp/atomic/atomic/operator_arith
	//       _current_queue_idx++ はatomic にpost-inc して修正前の値を返すようなので、
	//       同時にqueue_node() しても別々の要素にアクセスするのでok.
	//       -> いや、っていうかそもそも、同じSearcher オブジェクトのqueue_node() は並列に呼ばれないので、
	//          _current_queue_idx は普通のint型 にして問題ない。
	//std::cout << "info string queue_node(), _current_queue_idx = " << _current_queue_idx << std::endl;
	//std::cout << "info string queue_node(), _eval_queue.size() = " << _eval_queue.size() << std::endl;
	auto& queue_top = _eval_queue[_current_queue_idx++];
	queue_top.first = node;                               // set UctNode
	board.make_input_features(queue_top.second.first);    // set feature
	queue_top.second.second = board.turn();               // set color

	//// debug
	//if (node == global_tree->current_head) {
	//	std::cout << "info: N_FEATURE1_CHANNEL = " << N_FEATURE1_CHANNEL << std::endl;
	//	std::cout << "info: N_FEATURE2_CHANNEL = " << N_FEATURE2_CHANNEL << std::endl;
	//	std::cout << "========== feature1 ==========" << std::endl;
	//	for (int i = 0; i < N_FEATURE1_WHC; ++i) {
	//		if (i % N_FEATURE_WIDTH == 0) {
	//			std::cout << "\n";
	//		}
	//		if (i % N_FEATURE_WH == 0) {
	//			std::cout << "\n";
	//			std::cout << "channel = " << (i / N_FEATURE_WH) + 1 << std::endl;
	//		}

	//		std::cout << *(std::get<0>(queue_top.second.first) + i) << " ";
	//	}

	//	std::cout << "========== feature2 ==========" << std::endl;
	//	for (int i = 0; i < N_FEATURE2_WHC; ++i) {
	//		if (i % N_FEATURE_WIDTH == 0) {
	//			std::cout << "\n";
	//		}
	//		if (i % N_FEATURE_WH == 0) {
	//			std::cout << "\n";
	//			std::cout << "channel = " << (i / N_FEATURE_WH) + 1 << std::endl;
	//		}

	//		std::cout << *(std::get<1>(queue_top.second.first) + i) << " ";
	//	}
	//}

}

// queue の中身を一斉に評価
// NOTE
//     : 実装上thread safe のはず
//         : 同じオブジェクトからは一つだけ、異なるオブジェクトから並列で呼ぶ場合、
//           同じノードが異なるthread にqueue されていない(任意のnode のqueue される合計回数が1回である)
//           ことが保証されるならば、thread safe。
//             : node への書き込みが4件ある
void ParallelPuctSearcher::eval_node() {
	_parent_parallel_puct_searcher_group->nn_forward(_current_queue_idx, _input, _output_policy, _output_value);

	for (int i = 0; i < _current_queue_idx; ++i) {
		auto&& e = _eval_queue[i];
		auto node = e.first;
		auto&& color = e.second.second;

		// debug
		//std::cout << "info: _output_value[i] = " << _output_value[i] << std::endl;

#if defined (FEATURE_V2)
		// NOTE
		//     : dlshogi のmodel は、onnx に変換するときはvalue の出力にsimoid を適用しているのでここでは不要。
		node->value = _output_value[i];
#else
		node->value = sigmoid(_output_value[i]);    // ==== write ===== (恐らく、同じノードがqueue されることは無いのでok)
#endif

		// debug
		//sync_cout << "info: sigmoid(_output_value[i]) = " << sigmoid(_output_value[i]) << ",node->value = " << node->value << sync_endl;

		for (int j = 0; j < node->legal_moves_size; ++j) {
			const int& label = move_to_label(node->child_infos[j].move, color);
			//const int& label_dlshogi = __dlshogi_make_move_label(node->child_infos[j].move.value(), color);
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
			node->child_infos[j].probability = logit;    // ==== write ===== (恐らく、同じノードがqueue されることは無いのでok)

			//// debug
			//if (g_playout_count == 125) {

			//}

			// debug
#ifdef DEBGU_PUCT_20231125_2_
			if (i == 32) {
				std::cout << "info: [" << j << "] " << node->child_infos[j].move.toUSI() << "," << label << "," << label_dlshogi << "," << logit << std::endl;
				if (label != label_dlshogi) {
					std::cout << "Error!: label mismatch" << std::endl;
				}
			}
#endif

			//// debug
			//sync_cout
			//	<< "info: [" << j << "] "
			//	<< "logit = " << logit
			//	<< ",node->child_infos[j].probability = " << node->child_infos[j].probability
			//	<< sync_endl;
		}
		// softmax
		softmax_with_temperature(node, g_softmax_temperature);    // ==== write ===== (恐らく、同じノードがqueue されることは無いのでok)

		// debug
#ifdef DEBGU_PUCT_20231125_2_
		if (i == 32) {
			for (int j = 0; j < node->legal_moves_size; ++j) {
				std::cout << "info: [" << j << "] " << node->child_infos[j].move.toUSI() << "," << node->child_infos[j].probability << std::endl;
		    }
		}
#endif


		//sync_cout << "[" << g_softmax_temperature_inner << "]" << sync_endl;



		// NOTE
		//     : ここに到達して、ようやくこのノードより先を展開できる。(∵policy が設定されてようやくucb スコアが計算出来る。)
		node->set_evaled();    // ==== write ===== (恐らく、同じノードがqueue されることは無いのでok)
	}

	//_eval_queue.clear();    // NOTE: これしたら予めset しておいたfeature のアドレス潰すからダメでしょ...?
	_current_queue_idx = 0;
}

// ==================================================
// ParallelPuctPlayer
// ==================================================
// TODO
//     : 抽象クラスというか、基底クラスを作る。
//       ほんでもって、override 的なことして、基底のPlayerとしてオブジェクトは持ってるけど、
//       random_player のを呼び出す、的なのをしたい。
//     : なんか、header の中身変えただけだと、.exe に反映されん。

// NOTE
//     : ParallelPuctPlayer のメンバ関数は並列で複数呼んではならない。

// HACK: うーん、ちょっと名前長い気もする。
// go 自体の終了。
// 実行中ならstop して、join する。
void ParallelPuctPlayer::_stop_and_join_go_if_running() {
	// https://cpprefjp.github.io/reference/thread/thread/joinable.html
	if (_go_thread.joinable()) {    // 実行中である(detach されてないものとする)
		_set_go_stop(true);
		_stop_searcher();

		_go_thread.join();
	}
}

// NOTE: 全ての探索を停止して良い時に実行する。
// 全ての探索を中止する。(ただ中止せよという命令を各group に送るだけ)
// false からtrue へと遷移した場合、
// ここで初めて停止命令が下ったことになるので、group のstop を呼んでやる。
void ParallelPuctPlayer::_stop_searcher() {
	_searcher_stop.store(true);

	_searcher_group->stop();
	_root_dfpn_group->stop();
	_pv_dfpn_group->stop();
	_pv_dfpn_group_2->stop();
}

ParallelPuctPlayer::ParallelPuctPlayer()
	: BasePlayer()
{
	_position_str = "";

	_searcher_group = std::make_unique<ParallelPuctSearcherGroup>();
	_root_dfpn_group = std::make_unique<ParallelDfpnGroup>();
	_pv_dfpn_group = std::make_unique<ParallelPvDfpnGroup>();
	_pv_dfpn_group_2 = std::make_unique<ParallelPvDfpnGroup>();

	_self_TT = new ns_dfpn::TranspositionTable;
	_enem_TT = new ns_dfpn::TranspositionTable;

	set_g_is_pondering(false);

	// global
	global_tree = std::make_unique<UctTree>();
	g_playout_count = 0;
	g_last_pv_print_playout = 0;
	g_time_limit = 0;
	g_n_extend_time = 0;

	// USI option
	g_usi_ponder = true;
	g_stochastic_ponder = true;
	g_is_debug = false;
	g_n_threads = 5;
	g_batch_size = 128;
	g_minimum_thinking_time = 1750;
	g_minimum_thinking_time_ratio_x100 = 50;
	g_network_delay = 1000;
	g_div_init = 43;
	g_div_final = 13;
	g_extend_time_sf_ratio_x100 = 97;

	g_const_playout = 0;
	g_print_pv_per_playout = 50000;
	g_puct_c_base = 28288;
	g_puct_c_init_x100 = 144;
	g_puct_c_fpu_reduction_x100 = 27;
	g_puct_c_base_root = 25617;
	g_puct_c_init_root_x100 = 116;
	g_puct_c_fpu_reduction_root_x100 = 0;
	g_softmax_temperature_x100 = 174;
	g_draw_value_black_x1000 = 500;
	g_draw_value_white_x1000 = 500;
	//g_init_value_x100 = 0;    // TODO: 現在未使用。削除すべきかも。
	g_is_fp16 = true;
	g_onnx_model = "model.onnx";
	g_own_book = true;
	g_book_filename = "book/standard_book.db";
	g_book_moves = 32;
	g_book_black_eval_limit = 1;
	g_book_white_eval_limit = -80;
	g_book_depth_limit = 0;
	g_book_check_depth_limit_last = true;

	g_draw_ply = 512;
	g_dfpn_self_hash_size_mb = 32768;
	g_dfpn_enem_hash_size_mb = 32768;
	g_root_dfpn_n_threads = 4;
	g_root_dfpn_max_depth = 51;
	g_root_dfpn_max_node = INT_MAX;    // TODO: 0 も指定できるようにした方が良いな。0 で無限、みたいなさ。
	g_pv_dfpn_n_threads = 16;
	g_pv_dfpn_n_threads_per_node = 1;
	g_pv_dfpn_max_depth = 31;
	g_pv_dfpn_max_node = 300000;
}

ParallelPuctPlayer::~ParallelPuctPlayer() {
	delete _self_TT;
	delete _enem_TT;
}

void ParallelPuctPlayer::usi_loop() {
	// NOTE: 任意のコマンドを処理する節は、その節を抜けるまで、次のコマンドが処理されることはない。
	//       当然_go_thread の実行中は他のコマンドが処理される可能性があるし処理できるが、
	//       その手前まではシリアルに逐次的に実行される。(なので安心して。)
	while (true) {
		std::string cmd, head_token;
		getline(std::cin, cmd);
		const auto&& tokens = tokenize(cmd);
		if (tokens.size() > 0) {
			head_token = tokens[0];
		}
		else {
			head_token = "";
		}

		if (head_token == "usi") {
			usi();
		}
		else if (head_token == "mode") {    // usi コマンドでコメント、みたいな感じで表示できない？
			std::cout << "info string feature = " << get_feature_mode() << std::endl;
			std::cout << "info string dfpn = " << get_dfpn_mode() << std::endl;
#if defined(IS_DENRYUSEN_PLY_OFFSET)
			std::cout << "info string IS_DENRYUSEN_PLY_OFFSET" << std::endl;
#endif
		}
		else if (head_token == "setoption") {
			setoption(cmd.substr(10));
		}
		else if (head_token == "isready") {
			isready();
		}
		else if (head_token == "usinewgame") {
			usinewgame();
		}
		else if (head_token == "position") {
			position(cmd.substr(9));
		}
		else if (head_token == "quit") {
			_stop_and_join_go_if_running();
			exit(1);
		}
		else if (head_token == "stop") {
			_stop_and_join_go_if_running();
		}
		else if (head_token == "gameover") {
			_stop_and_join_go_if_running();
		}
		else if (head_token == "go") {
			//// 1) 今動いている探索を停止させる

			// TODO: ここでは_go_stop = true にせんでもええ気がすんやけどどうやろうか。
			//       だって、_go_stop が実際に用いられるのは、USI_ponder=true の場合(即ち対局の時)。
			//       具体的には、bestmove を返した直後で、_go_stop の値によって確率的ponder をするか否かが変わる。
			//       なので、ここに居る時に、"go が実行中 && bestmove をまだ返していない" という状況は起きえないはず。
			//       何故なら、"go が実行中 && bestmove をまだ返していない" ということは、
			//       go を一度送ったのに、bestmove を待たずに再度go を送ったということであり、これは対局中のGUI の合法な状態遷移 とは異なり、
			//       理想的なGUI(合法なオートマトン) では発生し得ない。
			//       -> あなた(数分前の私)のおっしゃる通りです。ここではそのフラグは値を弄る必要がありません。
			//          (となると、その次の初期化によるfalse も無くせて一石二鳥)
			//          ただ、関数だけどうするか。別々に似たような処理の関数を二つ用意するのが無難だろうか。
			//       -> でもあれじゃね、今はgo_stop を一度しか使わないからこれでええけど、
			//          bestmove 返した後に2回以上go_stop 使うならダメ。
			//          だって、ここでも一切に思考を終了してgo() を抜けて欲しいんだもの。
			// 別でgo が動いている状態で、もう一つgo の処理を始めてはならない。(go() を並列で実行してはならない。)
			stopwatch_t stop_go_thread_sw;
			stop_go_thread_sw.start();
			_stop_and_join_go_if_running();
			stop_go_thread_sw.stop();
#if defined(VERBOSE)
			sync_cout << "info string _stop_go_thread_if_running() done, time=" << int(stop_go_thread_sw.elapsed_ms()) << "ms" << sync_endl;
#endif
			//// 2) 初期化

			_set_go_stop(false);

			// TODO
			//     : ponder で例えばg_time_limit とかg_n_extend_time とかはshould_exit() でアクセスするので、
			//       データ競合を避けるために先にponder(_go_thread) を止めるべきでは？

			// HACK
			//     : go コマンドが来て即座にstop が送られてきた場合、
			//       _go_thread = std::thread(// 略 //)　に達するまでに_stop_go_thread_if_running() が呼ばれてしまう可能性があり、
			//       そうなるとそのstop は無効となってしまう。
			//       この処理全体をスレッドかすべきかもしれない。

			//// parse
			// TODO
			//     : フツーに以下のparser でgo も見てやる方が綺麗かも...?
			// https://zenn.dev/yohhoy/articles/quiz-init-vector
			const std::vector tokens_without_go(tokens.begin() + 1, tokens.end());

			peak_parser_t parser(tokens_without_go);
			g_go_info.reset();

			while (parser.peak_next_if_exists()) {
				const auto& pt = parser.get_peak_token();
				if (pt == "infinite") {
					g_go_info.is_infinite = true;
				}
				else if (pt == "ponder") {    // 現状は確率的ponder のみの実装なので使わない。
					g_go_info.is_ponder = true;
				}
				else if (pt == "btime") {
					parser.peak_next_expected_to_exist();
					parser.store_peak_token(g_go_info.time[Black]);
				}
				else if (pt == "wtime") {
					parser.peak_next_expected_to_exist();
					parser.store_peak_token(g_go_info.time[White]);
				}
				else if (pt == "binc") {
					parser.peak_next_expected_to_exist();
					parser.store_peak_token(g_go_info.inc[Black]);
				}
				else if (pt == "winc") {
					parser.peak_next_expected_to_exist();
					parser.store_peak_token(g_go_info.inc[White]);
				}
				else if (pt == "byoyomi") {
					parser.peak_next_expected_to_exist();
					parser.store_peak_token(g_go_info.byoyomi);
				}
			}

			if (!g_go_info.check_state()) {
				sync_cout << "info string Error: invalid go cmd = [" << cmd << "]" << sync_endl;
			}
			else {
				//// 持ち時間制御
				// HACK: ここでset_position() して、更にgo_impl() でもset_position_and_get_components() していて無駄。
				//       だけど、正しい現在のturn を分かりやすくバグを増やさず確実に取得するには、これしかなかった。。。
				//       (いや、まぁtoken数 を数えたら行けるんやけど、超絶安全ではないから。。。)
				// https://zenn.dev/reputeless/books/standard-cpp-for-competitive-programming/viewer/library-algorithm
				g_board.set_position(_get_position_str_safe());
				const Color& turn = g_board.turn();
				//const int& n_self_ply_total = g_board.ply() / 2;    // 今までに自分が指した手の数 (<- TODO: ほんまか？ちゃうと思うぞこれ。)
				//const int& div = std::max({
				//    div_init - (n_self_ply_total - 1) * 2,
				//	  div_second - (n_self_ply_total - n_self_ply_while_begining - 1),
				//    iv_final
				// });

				const int& div = std::max({
					g_div_init - g_board.ply(),
					g_div_final
				});

				const int& base_time = (g_go_info.get_time(turn) / div);
				if (g_go_info.is_infinite || g_go_info.is_ponder) {    // 時間制限は無い
					g_time_limit = GoInfo::TIME_NONE;
				}
				else {
					if (g_go_info.is_inc()) {
						g_time_limit = base_time + g_go_info.get_inc(turn) - g_network_delay;
					}
					else if (g_go_info.is_byoyomi()) {
						g_time_limit = base_time + g_go_info.byoyomi - g_network_delay;
					}
					else if (g_go_info.is_extra_time_zero(turn)) {
						// HACK: 一応0でclip してあるけど、こうなったら正直投了では？ただ、切れ負け自体やること無いと思うので適当に。
						g_time_limit = my_max(base_time - g_network_delay, 0);
					}
					else {
						sync_cout << "info string Error: got unexpected instruction about time." << sync_endl;
						throw std::runtime_error("Error");
					}
				}

				g_n_extend_time = 0;

				// debug
				//sync_cout << "info string gi = " << g_go_info.to_str() << sync_endl;
				sync_cout << "info string time_limit = " << g_time_limit << ", div = " << div << ", ply = " << g_board.ply() << sync_endl;

				// https://tadaoyamaoka.hatenablog.com/entry/2018/02/20/223237
				_go_thread = std::thread([this]() {
					this->go();
					});
			}

		}
		else if (cmd == "") {
			// ただの改行ならスルー(getline は\n は削除して渡してくるはず)
		}
		else {
			std::cout << "ParserError: parser got unexpected cmd == [" << cmd << "]" << std::endl;;
			//throw std::runtime_error("Error");
		}
	}
}

void ParallelPuctPlayer::usi() {
	// name, author
	std::cout << "id name ParallelPuctPlayerV0.2.7" << std::endl;
	std::cout << "id author lefu777" << std::endl;

	// check, 時間制御, 探索 等
	std::cout << "option name " << "USI_Ponder" << " type " << "check" << " default " << bts(g_usi_ponder) << std::endl;
	std::cout << "option name " << "stochastic_ponder" << " type " << "check" << " default " << bts(g_stochastic_ponder) << std::endl;
	std::cout << "option name " << "is_debug" << " type " << "check" << " default " << bts(g_is_debug) << std::endl;
	std::cout << "option name " << "minimum_thinking_time" << " type " << "spin" << " default " << g_minimum_thinking_time << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "minimum_thinking_time_ratio_x100" << " type " << "spin" << " default " << g_minimum_thinking_time_ratio_x100 << " min " << 0 << " max " << 100 << std::endl;
	std::cout << "option name " << "network_delay" << " type " << "spin" << " default " << g_network_delay << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "div_init" << " type " << "spin" << " default " << g_div_init << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "div_final" << " type " << "spin" << " default " << g_div_final << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "extend_time_sf_ratio_x100" << " type " << "spin" << " default " << g_extend_time_sf_ratio_x100 << " min " << 0 << " max " << INT_MAX << std::endl;

	std::cout << "option name " << "threads" << " type " << "spin" << " default " << g_n_threads << " min " << 1 << " max " << 16 << std::endl;
	std::cout << "option name " << "batch_size" << " type " << "spin" << " default " << g_batch_size << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "const_playout" << " type " << "spin" << " default " << g_const_playout << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "print_pv_per_playout" << " type " << "spin" << " default " << g_print_pv_per_playout << " min " << 0 << " max " << INT_MAX << std::endl;

	// puct パラメータ
	std::cout << "option name " << "puct_c_base" << " type " << "spin" << " default " << g_puct_c_base << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "puct_c_init_x100" << " type " << "spin" << " default " << g_puct_c_init_x100 << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "puct_c_fpu_reduction_x100" << " type " << "spin" << " default " << g_puct_c_fpu_reduction_x100 << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "puct_c_base_root" << " type " << "spin" << " default " << g_puct_c_base_root << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "puct_c_init_root_x100" << " type " << "spin" << " default " << g_puct_c_init_root_x100 << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "puct_c_fpu_reduction_root_x100" << " type " << "spin" << " default " << g_puct_c_fpu_reduction_root_x100 << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "softmax_temperature_x100" << " type " << "spin" << " default " << g_softmax_temperature_x100 << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "draw_value_black_x1000" << " type " << "spin" << " default " << g_draw_value_black_x1000 << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "draw_value_white_x1000" << " type " << "spin" << " default " << g_draw_value_white_x1000 << " min " << 0 << " max " << INT_MAX << std::endl;
	//std::cout << "option name " << "init_value_x100" << " type " << "spin" << " default " << g_init_value_x100 << " min " << 0 << " max " << INT_MAX << std::endl;

	// model, book
	std::cout << "option name " << "is_fp16" << " type " << "check" << " default " << bts(g_is_fp16) << std::endl;
	std::cout << "option name " << "onnx_model" << " type " << "string" << " default " << g_onnx_model << std::endl;
	std::cout << "option name " << "own_book" << " type " << "check" << " default " << bts(g_own_book) << std::endl;
	std::cout << "option name " << "book_file" << " type " << "string" << " default " << g_book_filename << std::endl;
	std::cout << "option name " << "book_moves" << " type " << "spin" << " default " << g_book_moves << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "book_black_eval_limit" << " type " << "spin" << " default " << g_book_black_eval_limit << " min " << INT_MIN << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "book_white_eval_limit" << " type " << "spin" << " default " << g_book_white_eval_limit << " min " << INT_MIN << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "book_depth_limit" << " type " << "spin" << " default " << g_book_depth_limit << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "book_check_depth_limit_last" << " type " << "check" << " default " << bts(g_book_check_depth_limit_last) << std::endl;

	// dfpn
	std::cout << "option name " << "draw_ply" << " type " << "spin" << " default " << g_draw_ply << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "dfpn_self_hash_size" << " type " << "spin" << " default " << g_dfpn_self_hash_size_mb << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "dfpn_enem_hash_size" << " type " << "spin" << " default " << g_dfpn_enem_hash_size_mb << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "root_dfpn_threads" << " type " << "spin" << " default " << g_root_dfpn_n_threads << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "root_dfpn_max_depth" << " type " << "spin" << " default " << g_root_dfpn_max_depth << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "root_dfpn_max_node" << " type " << "spin" << " default " << g_root_dfpn_max_node << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "pv_dfpn_threads" << " type " << "spin" << " default " << g_pv_dfpn_n_threads << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "pv_dfpn_threads_per_node" << " type " << "spin" << " default " << g_pv_dfpn_n_threads_per_node << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "pv_dfpn_max_depth" << " type " << "spin" << " default " << g_pv_dfpn_max_depth << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "option name " << "pv_dfpn_max_node" << " type " << "spin" << " default " << g_pv_dfpn_max_node << " min " << 0 << " max " << INT_MAX << std::endl;
	std::cout << "usiok" << std::endl;
}

// HACK: これ、流石にUSI_options とか作って自動でparse するようにせんと。。。
// http://shogidokoro.starfree.jp/usi.html
// @argcmd
//     : ex>
//       "name threads value 4"
void ParallelPuctPlayer::setoption(std::string cmd) {
	const auto& tokens = tokenize(cmd);
	if (tokens.size() < 4) {
		std::cout << "Warning: the number of tokens in the setoption command must be equal to or greater than 4." << std::endl;
		return;
	}

	if (tokens[1] == "is_debug") {
		g_is_debug = stb(tokens[3]);
	}
	else if (tokens[1] == "USI_Ponder") {
		g_usi_ponder = stb(tokens[3]);
	}
	else if (tokens[1] == "stochastic_ponder") {
		g_stochastic_ponder = stb(tokens[3]);
	}
	else if (tokens[1] == "minimum_thinking_time") {
		g_minimum_thinking_time = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "minimum_thinking_time_ratio_x100") {
		g_minimum_thinking_time_ratio_x100 = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "network_delay") {
		g_network_delay = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "div_init") {
		g_div_init = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "div_final") {
		g_div_final = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "extend_time_sf_ratio_x100") {
		g_extend_time_sf_ratio_x100 = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "threads") {
		g_n_threads = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "batch_size") {
		g_batch_size = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "const_playout") {
		g_const_playout = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "print_pv_per_playout") {
		g_print_pv_per_playout = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "puct_c_base") {
		g_puct_c_base = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "puct_c_init_x100") {
		g_puct_c_init_x100 = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "puct_c_fpu_reduction_x100") {
		g_puct_c_fpu_reduction_x100 = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "puct_c_base_root") {
		g_puct_c_base_root = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "puct_c_init_root_x100") {
		g_puct_c_init_root_x100 = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "puct_c_fpu_reduction_root_x100") {
		g_puct_c_fpu_reduction_root_x100 = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "softmax_temperature_x100") {
		g_softmax_temperature_x100 = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "draw_value_black_x1000") {
		g_draw_value_black_x1000 = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "draw_value_white_x1000") {
		g_draw_value_white_x1000 = std::atoi(tokens[3].c_str());
	}
	//else if (tokens[1] == "init_value_x100") {
	//	g_init_value_x100 = std::atoi(tokens[3].c_str());
	//}
	else if (tokens[1] == "is_fp16") {
		g_is_fp16 = stb(tokens[3]);
	}
	else if (tokens[1] == "onnx_model") {
		g_onnx_model = tokens[3];
	}
	else if (tokens[1] == "own_book") {
		g_own_book = stb(tokens[3]);
	}
	else if (tokens[1] == "book_file") {
		g_book_filename = tokens[3];
	}
	else if (tokens[1] == "book_moves") {
		g_book_moves = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "book_black_eval_limit") {
		g_book_black_eval_limit = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "book_white_eval_limit") {
		g_book_white_eval_limit = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "book_depth_limit") {
		g_book_depth_limit = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "book_check_depth_limit_last") {
		g_book_check_depth_limit_last = stb(tokens[3]);
	}
	else if (tokens[1] == "draw_ply") {
		g_draw_ply = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "dfpn_self_hash_size") {
		g_dfpn_self_hash_size_mb = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "dfpn_enem_hash_size") {
		g_dfpn_enem_hash_size_mb = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "root_dfpn_threads") {
		g_root_dfpn_n_threads = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "root_dfpn_max_depth") {
		g_root_dfpn_max_depth = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "root_dfpn_max_node") {
		g_root_dfpn_max_node = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "pv_dfpn_threads") {
		g_pv_dfpn_n_threads = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "pv_dfpn_threads_per_node") {
		g_pv_dfpn_n_threads_per_node = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "pv_dfpn_max_depth") {
		g_pv_dfpn_max_depth = std::atoi(tokens[3].c_str());
	}
	else if (tokens[1] == "pv_dfpn_max_node") {
		g_pv_dfpn_max_node = std::atoi(tokens[3].c_str());
	}
	else {
		std::cout << "info string Error: got unexpected setoption name = " << tokens[1] << std::endl;
	}
}

bool ParallelPuctPlayer::_isready() {

	//// ------------------------------
	//// 1) 構成ファイルがあるなら、そこからoption を読み込んでset
	//// ------------------------------

	// 現状、以下をset する想定。それ以外をset した場合の挙動は保証しない。
	//   : C_init
    //   : C_base
    //   : C_fpu_reduction
    //   : C_fpu_reduction_root
    //   : C_init_root
    //   : C_base_root
    //   : Softmax_Temperature
	Compiler::LLParser parser(g_onnx_model + ".ini");
	if (parser.get_input_size() > 0) {    // 何かしらファイルがあって、且つ中身が空ではない
		parser.compile();
		parser.print_name_table();

		if (!parser.g_error_is_ok()) {    // 何かしら構文にエラーがあったなら、構成ファイルの内容は使わない。
			parser.g_error_flush_with_header("info string ");
		}
		else{    // 但しくparse 出来たなら、構成ファイルの内容に基づいてoption をset
			auto name_table = parser.get_name_table();
			for (auto&& entry : name_table) {
				// オプション名がdlshogi での名前ならば、こっちでの名前に変換。
				auto it = option_name_from_dlshogi.find(entry.get_name());
				if (it != option_name_from_dlshogi.end()) {
					entry.set_name(it->second);
				}

				std::string cmd = join(" ", "name", entry.get_name(), "value", std::to_string(entry.get_value()));
				setoption(cmd);
			}
		}
	}

	//// ------------------------------
	//// 2) 値が小数であるUSI_option のset, 0超過であるべきオプションが0超過であることの確認。
	//// ------------------------------
	
	// TODO
	//     : やるとしたら、ここで.ini のparse
	//       dlshogi 形式の.ini も読み込めるように。(オプション名がdlshogi の物でも読み込めるように。)

	// HACK: 0超過であることは、setoption 時に確認して、以下ならその場で警告すべきだよね。
	//       (いやまぁ本当は0 入力を制限すりゃええんやけど、入力するときに勝手に値がクリップされたりして面倒やし。。。)
	if (g_div_init <= 0) {
		std::cout << "info string Error!: g_div_init must be greater than 0." << std::endl;
		return false;
	}

	if (g_div_final <= 0) {
		std::cout << "info string Error!: g_div_final must be greater than 0." << std::endl;
		return false;
	}

	if (g_extend_time_sf_ratio_x100 <= 0) {
		std::cout << "info string Error!: g_extend_time_sf_ratio_x100 must be greater than 0." << std::endl;
		return false;
	}
	g_extend_time_sf_ratio = g_extend_time_sf_ratio_x100 / 100.0f;

	if (g_pv_dfpn_n_threads_per_node <= 0) {
		std::cout << "info string Error!: g_pv_dfpn_n_threads_per_node must be greater than 0." << std::endl;
		return false;
	}

	// WARNING
	//     : g_softmax_temperature_x100 等を100 で割る時は、float型の100.0f で割る事！！
	// USI でUSI_option を入力する時、0 も入力可能にしておいた方が、
	// 値を変更するときに(勝手に最小値でclip されないので)便利 なので、
	// 0 も入力可能にしてあるが、実際に0 が入力された場合は弾く。
	if (g_softmax_temperature_x100 <= 0) {
		std::cout << "info string Error!: g_softmax_temperature_x100 must be greater than 0." << std::endl;
		return false;
	}
	g_softmax_temperature = g_softmax_temperature_x100 / 100.0f;

	if (g_puct_c_init_x100 <= 0) {
		std::cout << "info string Error!: g_puct_c_init_x100 must be greater than 0." << std::endl;
		return false;
	}
	g_puct_c_init = g_puct_c_init_x100 / 100.0f;

	if (g_puct_c_init_root_x100 <= 0) {
		std::cout << "info string Error!: g_puct_c_init_root_x100 must be greater than 0." << std::endl;
		return false;
	}
	g_puct_c_init_root = g_puct_c_init_root_x100 / 100.0f;

	g_puct_c_fpu_reduction = g_puct_c_fpu_reduction_x100 / 100.0f;
	g_puct_c_fpu_reduction_root = g_puct_c_fpu_reduction_root_x100 / 100.0f;
	//g_init_value = g_init_value_x100 / 100.0f;
	g_minimum_thinking_time_ratio = g_minimum_thinking_time_ratio_x100 / 100.0f;

	g_draw_value_black = g_draw_value_black_x1000 / 1000.0f;
	g_draw_value_white = g_draw_value_white_x1000 / 1000.0f;

	//// ------------------------------
	//// 3) 置換表のallocate とclear
	//// ------------------------------

	_self_TT->Resize(g_dfpn_self_hash_size_mb);
	_enem_TT->Resize(g_dfpn_enem_hash_size_mb);
	_self_TT->Clear();
	_enem_TT->Clear();

	//// ------------------------------
	//// 4) thread group の初期化
	//// ------------------------------

	//// searcher_group

	if (!_searcher_group->_isready()) {
		std::cout << "info string Error!: failed to ready _searcher_group." << std::endl;
		return false;
	}

	//// root dfpn
	
	_root_dfpn_group->init(
		g_root_dfpn_n_threads, g_root_dfpn_max_depth,
		g_root_dfpn_max_node, g_draw_ply
	);    // root dfpn は、毎root 局面ごとに対応する置換表を渡すので、ここでは渡さない。

	//// pv dfpn

	_pv_dfpn_group->init(
		_self_TT, _enem_TT,
		g_pv_dfpn_n_threads, g_pv_dfpn_n_threads_per_node,
		g_pv_dfpn_max_depth, g_pv_dfpn_max_node, g_draw_ply
	);    // pv dfpn は、毎root 局面ごとに置換表を2つとも使うことになるので、事前に二つとも渡しておく。

	_pv_dfpn_group_2->init(
		_self_TT, _enem_TT,
		g_root_dfpn_n_threads, g_pv_dfpn_n_threads_per_node,
		g_pv_dfpn_max_depth, g_pv_dfpn_max_node, g_draw_ply
	);    // root dfpn が終了次第、そこで使ってた分のスレッドも丸々PvDfpn に回すので、g_root_dfpn_n_threads。


	//// ------------------------------
	//// 5) その他
	//// ------------------------------

	//// 局面の初期化
	g_board.reset();

	//// 電竜戦に対応
#if defined(IS_DENRYUSEN_PLY_OFFSET)
	g_div_init += DENRYUSEN_PLY_OFFSET;
	g_book_moves += DENRYUSEN_PLY_OFFSET;
#endif

	//// book
	if (g_own_book) {    // own_book が有効な場合のみ、定跡を読み込む
		if (!_yobook_parser.parse(g_book_filename)) {
			std::cout << "info string Error!: failed to parse book, so start without a book. " << std::endl;
		}
		else {
			std::cout << "info string _yobook_parser.size() = [" << _yobook_parser.size() << "]" << std::endl;
		}
	}

	return true;
}

void ParallelPuctPlayer::isready() {
	if (_isready()) {
		sync_cout << "readyok" << sync_endl;
	}
	else {
		sync_cout << "info string Error!: ParallelPuctPlayer::isready() failed." << sync_endl;
		throw std::runtime_error("Error");
	}
}

// @arg moves
//     : 以下の2通りがある。
//       startpos moves xxxx xxxx ...
//       [sfen形式の任意の初期局面(先頭にsfenは無い)] moves xxxx xxxx ...
void ParallelPuctPlayer::position(const std::string& position_str) {
	//std::cout << "info got cmd = position" << std::endl;
	//if (!_board.set_position(moves)) {
	//	std::cout << "info Error: failed to set_position" << std::endl;
	//}
	_set_position_str_safe(position_str);
}


// https://github.com/TadaoYamaoka/python-dlshogi/blob/master/pydlshogi/player/mcts_player.py
// TODO
//     : ただひたすらに探索木上の局面のentry を参照して、pn == 0 ならset_win() し続けるだけのスレッドがあっても良いのでは？
// NOTE
//     : この関数が呼ばれるまでに、g_board のset が終わっていなければならない。
//     : root_dfpnのrun() join() と、searcher のrun() join() がそれぞれ連続して実行している箇所が必要。
//       (root_dfpn と、searched(puct) の結果をパラレルに取得できるようにしたい。)
//     : 探索の中止には以下の23パターンがある。但し、中止とは_stop_searcher() を呼ぶことを指す。
//           : puct 探索が終了したら、問答無用で探索を中止する。
//           : root で詰ます手が見つかれば、問答無用で中止する。
//           : USI 側からstop が来れば、問答無用で中止する(usi_loop() にて)。
//     : まだroot の詰み探索が完了してない状況で、USI 側のstop コマンドが来て、その直後にroot で詰みが見つかった場合、
//       dfpn による詰ます手を指して欲しいので、以下のような実装をしている。
// puct での探索を実行。
// 終局 or 宣言勝ち の局面では呼び出さない。
// TODO
//     : std::pair<Move, Move> を返すように。(それぞれ、bestmove, pondermove)
std::pair<Move, Move> ParallelPuctPlayer::go_puct() {
	//// --------------------
	//// 1) 初期化等
	//// --------------------

	assert(!(g_board.is_game_over() || g_board.is_nyugyoku()));

	// プレイアウト数をクリア
	g_playout_count = 0;
	g_last_pv_print_playout = 0;

	// ルートノードが未展開なら展開する
	if (!global_tree->current_head->is_expanded()) {    // 未展開
		global_tree->current_head->expand_node(g_board);
	}

	//// --------------------
	//// 2) 探索以外で手を返せるなら返す。
	//// --------------------

	// 候補手が一つの場合は、その手を返す。
	__LegalMoveList legal_moves(g_board);
	if (legal_moves.size() == 1) {
		sync_cout << "info" << " pv " << legal_moves.Move().toUSI() << sync_endl;

		// TODO: ここでこの処理するのはワンちゃんバグるので要検証。
		//       -> 前に理論上は対応した。100戦はゆうに対局したが現状はバグっていない。
		Move bestmove;
		Move pondermove;
		float bestvalue;
		get_bestmove_and_print_pv(&bestmove, &pondermove, nullptr);

		assert(legal_moves.Move() == bestmove);

		return { legal_moves.Move(), pondermove};
	}

	// 序盤で定跡から指せるなら、定跡の手を指す。
	// 但し、ponder 中は定跡を用いず思考してもらう。
	if (!get_g_is_pondering()) {
		if (g_board.ply() <= g_book_moves) {
			const auto& find_status = _yobook_parser.find(g_board.toSFEN());
			if (find_status.first == true) {    // 定跡にhit
				const yo_book_value_t& book_hit = (*(find_status.second)).second;
				const int book_eval_limit = (g_board.turn() == Black ? g_book_black_eval_limit : g_book_white_eval_limit);
				const auto& find_status2= book_hit.get_best_eval_move(book_eval_limit, g_book_depth_limit, g_book_check_depth_limit_last);

				book_hit.print_all_moves();
				std::cout << "info string depth_limit=" << g_book_depth_limit << ", eval_limit=" << book_eval_limit << std::endl;

				if (find_status2.first) {
					const yo_book_move_t& bestmove = find_status2.second;

					bestmove.print_usi();
					return { bestmove.move1, bestmove.move2 };
				}
			}
		}
	}

	//// --------------------
	//// 3) 探索
	//// --------------------

	//// 初期化等

	_searcher_stop.store(false);
	_self_TT->NewSearch();
	_enem_TT->NewSearch();

	_root_dfpn_group->set_root_board(g_board);
	_root_dfpn_group->set_tt(_self_TT);
	_pv_dfpn_group->set_root_board(g_board);
	_pv_dfpn_group_2->set_root_board(g_board);

	bool is_root_mate = false;                 // 必ずthread_root_dfpn.join() の後にread すること。
	Move root_mate_move = Move::moveNone();    // 必ずthread_root_dfpn.join() の後にread すること。

	//// run

	std::thread thread_root_dfpn([this, &is_root_mate, &root_mate_move]() {
		    _root_dfpn_group->run();
			if (_searcher_stop.load()) {
				// NOTE: _root_dfpn_group の実行よりも前に_searcher_group が終了したことで、
				//       PUCT終了による探索中止が無視されたことの必要条件。
				_root_dfpn_group->stop();
			}
			_root_dfpn_group->join();
			
			sync_cout << "info string _root_dfpn_group.join() done." << sync_endl;

			__Board board_cpy(g_board);
			root_mate_move = _root_dfpn_group->get_move(board_cpy);
			if (_root_dfpn_group->get_is_mate()
				&& root_mate_move != Move::moveNone()    // 置換表衝突すると詰ます手が見れなくなるので念のため確認。
			) {
				is_root_mate = true;
				_stop_searcher();
			}
			else if (!_searcher_stop.load()) {    // 探索停止出ない場合(root dfpn が、全て探索しきって終了した場合)
				// FIXME
				//     : 1. root_dfpn 終了
				//       2. (_searcher_stop==false なので)
				//          print("pv_dfpn2 start")
				//       3. searcher_group 終了で、_stop_searcher()
				//       4. pv_dfpn2.run()
				//          -> stop() と入れ違いになっていて、終了しない。
#if defined(VERBOSE)
				sync_cout << "info string _pv_dfpn_group_2.run() start." << sync_endl;
#endif
				_pv_dfpn_group_2->run();
				// NOTE: run() 直前はstop==false だったが、run() してみるとどうやらstop==true だった場合、即座に停止命令を送る、
				//     : (1) stop==false を確認した後で、(2) 入れ違いでstop==true となって停止命令が送られ、(3) 更にその後でrun() しちゃった場合は、ここで止める。
				//       この入れ違いで問題なのは、"実行していない時のstop()" が無視されてしまうことである。
			    //       stop() が無視されてしまった可能性がある場合は、再度"実行中に"stop() を送る。
				//       (run() 直後のここでstop==true が観測されることは、stop() が無視されていることの必要条件である。)
				if (_searcher_stop.load()) {
					_pv_dfpn_group_2->stop();
				}
				_pv_dfpn_group_2->join();

#if defined(VERBOSE)
				sync_cout << "info string _pv_dfpn_group_2.join() done." << sync_endl;
#endif
			}
		}
	);
	_searcher_group->run();
	_pv_dfpn_group->run();
	if (_searcher_stop.load()) {
		// NOTE: _searcher_group, _pv_dfpn_group の実行 より前にroot_dfpn が詰みを見つけたことで、
		//       詰み発見による探索中止が無視されたことの必要条件。
		_searcher_group->stop();
		_pv_dfpn_group->stop();
	}
	else if (_go_stop.load()) {
		// NOTE: _searcher_group, _pv_dfpn_group, thread_root_dfpn(正確には_root_dfpn_group or _pv_dfpn_group_2) 
		//       の実行より前にGUI 側からstop() が送られて来たことで、その停止命令が無視されたことの必要条件。
		//       (通常の思考, 確率的ponder のどちらでも起こり得る。)
		_stop_searcher();
	}

	//// join

	_searcher_group->join();
	// NOTE: この時点で_searcher_group の終了は保証されている。root_dfpn の開始は保証されていない。
	_stop_searcher();

	thread_root_dfpn.join();
	_pv_dfpn_group->join();

#if defined(VERBOSE)
	sync_cout << "info string all join() done." << sync_endl;
#endif

	//// --------------------
	//// 4) 今回のstop が何に起因するかによって返す手を決める
	//// --------------------

	if (is_root_mate) {    // root_dfpn の詰み発見によるもの
		const auto& elapsed_time_ms = g_stop_watch.elapsed_ms_interim();
		const auto& nps = (1000 * _root_dfpn_group->get_searched_node()) / elapsed_time_ms;

		sync_cout
			<< "info"
			<< " score mate +"
			<< " pv " << root_mate_move.toUSI()
			<< " nodes " << _root_dfpn_group->get_searched_node()
			<< " nps " << positive_decimal_to_int_str(nps)
			<< " time " << positive_decimal_to_int_str(elapsed_time_ms)
			<< sync_endl;
		// 詰み発見したなら、別にponder 無くてもええやろ
		return { root_mate_move, Move::moveNone()};
	}
	else {    // puct が終了 || USI からのstop
		Move bestmove;
		Move pondermove;
		float bestvalue;
		get_bestmove_and_print_pv(&bestmove, &pondermove, nullptr);

		// TODO: ここでbestvalue の値によっては投了
		return { bestmove, pondermove};
	}
}

// TODO
//     : Move にresign とwin があるなら、コイツもMove で返したい。
//       現状、resign とwin に相当するMove は無いはずで、string で返さざるを得ない。
//       (有りそうなもんやけどな。探そうぜ。まぁ最悪実装してもええけどなんかバグりそうで怖かったりもする。)
//       (まぁ一応、通常はそのフラグは使わず、ここでだけ使うなら一応バグりはせんはずやけども。)
// 最善手を文字列で返す。
std::pair<std::string, std::string> ParallelPuctPlayer::go_impl(const std::string& position_str) {
    // for debug
	stopwatch_t reset_to_position_sw;

	std::string startsfen;
	std::vector<Move> moves;
	Key startpos_key;
	bool status = g_board.set_position_and_get_components(position_str, &startsfen, &moves, &startpos_key);
	assert(status && "failed to __Board::set_position_and_get_components()");

	reset_to_position_sw.start();
	bool found = global_tree->reset_to_position(startpos_key, moves);
	reset_to_position_sw.stop();

	// debug
	sync_cout
		<< "info string found_old_head = " << bts(found)
		<< ", tree.reset_to_position_time = " << int(reset_to_position_sw.elapsed_ms()) << "ms"
		<< sync_endl;
	if (reset_to_position_sw.elapsed_ms() > 2000) {
		flush_all_and_clear_debug_str();
		sync_cout << "info string Error!: reset_to_position_sw() is too long." << sync_endl;
		exit(1);
	}

	g_stop_watch.start();

	const auto&& mate_move3 = g_board.mateMove(3);

	std::string bestmove_usi = Move::moveNone().toUSI();
	std::string pondermove_usi = Move::moveNone().toUSI();

	if (g_board.is_game_over()) {
		bestmove_usi = MOVE_USI_RESIGN;
	}
	else if (g_board.is_nyugyoku()) {
		bestmove_usi = MOVE_USI_WIN;
	}
	else if (mate_move3) {
		bestmove_usi = __move_to_usi(mate_move3);
		sync_cout << "info score mate + pv " << bestmove_usi << sync_endl;
	}
	else {
		auto [bestmove, pondermove] = go_puct();

		bestmove_usi = bestmove.toUSI();
		pondermove_usi = pondermove.toUSI();

		if (g_is_debug) {
			//print_ucb(nullptr, global_tree->current_head);

		    //// debug
			if (global_tree->current_head->is_expanded() && global_tree->current_head->is_evaled()) {
				sync_cout << "info: [go() done] print_ucb(head)" << sync_endl;
				print_ucb(nullptr, global_tree->current_head);

				//sync_cout << "info: [go() done] print_ucb(head->debug_tag_idx_1)" << sync_endl;
				//const int debug_tag_idx_1 = 71;
				//UctNode* next_node_1 = print_child_and_transition(global_tree->current_head, debug_tag_idx_1);

				//if (next_node_1 != nullptr) {
				//	sync_cout << "info: [go() done] print_ucb(head->debug_tag_idx_1->debug_tag_idx_2)" << sync_endl;
				//	const int debug_tag_idx_2 = 84;
				//	UctNode* next_node_2 = print_child_and_transition(next_node_1, debug_tag_idx_2);
				//}
			}
		}
	}

	g_stop_watch.stop();

	return { bestmove_usi, pondermove_usi};
}

// TODO
//     : USI コマンドと入れ違いでfalse にしちゃうパターンとか、
//       run() 中にstop() しちゃうパターンとか、色々考えるとキリがない。。。
//     : _shared_stop を毎探索前にfalse にset したいが、usi との入れ違いが怖くて。。。
//       -> USI_ponder 使って、
//          go ponder の時はusi から渡されたposition の末尾の指し手を削ればok な希ガス。一番無難。
//              : go のparse 時にponder を追加。
//              : 一旦、usi に送るponder の指し手はrandom pick(or idx==0) の指し手を返せば良い気もする。
//                だって、仮にponder move がPV 上から得られなかった場合、
//                確率的ponder が動かない。
//                -> ponder が定まらなかったら適当にすれば良いと思うが、
//                   そうでないなら見てくれ的にちゃんとした手を選びたい。。。
//     : _shared_stop を毎探索前にfalse にset する。
//     : ここで、if(ponder) { // 最後のbestmove 表示とかは無しで、定跡も使わないで試行 } else { // これまで通り } とする。
//       又、go_parser だかなんだかで、ponder か否かのフラグを持つ。
//       (定跡使わない云々は、go_impl() で、go_info.ponder == true ならそうするように。)
//           : 雑にこうしたけど、ちゃんとponder の実装考える。
//     : global_goInfo を使える箇所は使う。
//     : root dfpn 終わったら、pv dfpn 動かすように。(_pv_dfpn_group_2)
//       thread_per_node はpv_dfpn と同じ値に。
//           : 次の計測ではこいつを測りたい。
//     : 現状、g_stochastic_ponder のフラグは無意味。問答無用で確率的ponder が実行される。
//           : 通常のponder を実装するなら、ponder の際は探索木の解放はしてはならない。
//             (ponder 外れた時、今までの探索結果は全部消えてることになる)
// NOTE
//     : 並列実行してはならない。(想定していない。)
void ParallelPuctPlayer::go() {
	//// 1) 通常の思考

	auto position_str = _get_position_str_safe();
	auto [bestmove_usi, pondermove_usi] = go_impl(position_str);

#if 0
	// TODO: 通常のponder が未実装なので、こうしたら良くない(g_stochastic_ponder==fales を指定した時、予期しない挙動となる。)
	if (g_usi_ponder && !g_stochastic_ponder && false) {
		if (pondermove_usi != Move::moveNone().toUSI()) {
			sync_cout << "bestmove " << bestmove_usi << " ponder " << pondermove_usi << sync_endl;
		}
		else {
			sync_cout << "bestmove " << bestmove_usi << sync_endl;
		}
	}
	else {
		sync_cout << "bestmove " << bestmove_usi << sync_endl;
	}
#else
	sync_cout << "bestmove " << bestmove_usi << sync_endl;
#endif

	//// 2) 確率的ponder

	if (!_get_go_stop()) {    // 関数を抜けずに、このまま処理を続けて良い
		if (g_usi_ponder && (true | g_stochastic_ponder)) {     // ponder が有効で、且つ確率的ponder である
			if (bestmove_usi != MOVE_USI_RESIGN && bestmove_usi != MOVE_USI_WIN) {    // さっきの指し手で対局は終了しない
				// 1手進める
				const Move& bestmove = g_board.Move_from_usi(bestmove_usi);
				g_board.push(bestmove);
				if (find_token(position_str, "moves")) {
					position_str = join(" ", position_str, bestmove_usi);
				}
				else {    // 対局開始局面
					position_str = join(" ", position_str, "moves", bestmove_usi);
				}

				set_g_is_pondering(true);
#if defined(VERBOSE)
				sync_cout
					<< "info"
					<< " depth --"
					<< " score cp -----"
					<< " nodes -----"
					<< " time -----"
					<< " pv --------------------------------------------------------------------------------"
					<< sync_endl;
#endif
				sync_cout << "info string stochastic_ponder start." << sync_endl;

				// NOTE: 候補手が1つしかない場合は探索が実行されずに終わってしまうので、
				//       次の局面で探索をする。(この時、通常のponder が思考する局面を思考するので、通常のponder と等価。)
				//     : ここで合法手生成するのは無駄(go_puct() でもする) だけど、go_impl() とかにこういう処理を入れたくなかった。
				//       彼には与えられた局面で思考することに専念して欲しかった。
				__LegalMoveList legal_move_list(g_board);
				if (legal_move_list.size() == 1) {
					// 更にもう1手進める
					const Move& enemy_bestmove = legal_move_list.Move();
					g_board.push(enemy_bestmove);
					position_str = join(" ", position_str, enemy_bestmove.toUSI());

					// debug
					//sync_cout << "info string stochastic_ponder start, pos=" << position_str << sync_endl;

					go_impl(position_str);
				}
				else {
					// debug
					//sync_cout << "info string stochastic_ponder start, pos=" << position_str << sync_endl;

					go_impl(position_str);
				}

				set_g_is_pondering(false);
#if defined(VERBOSE)
				sync_cout
					<< "info"
					<< " depth --"
					<< " score cp -----"
					<< " nodes -----"
					<< " time -----"
					<< " pv --------------------------------------------------------------------------------"
					<< sync_endl;
#endif
				sync_cout << "info string stochastic_ponder done." << sync_endl;
			}
		}
	}
}

// ==================================================
// debug utils
// ==================================================
UctNode* print_child_and_transition(UctNode* parent, const int child_idx) {
	if (parent->legal_moves_size > child_idx) {    // このindex はアクセス違反とならない。
		if (parent->child_infos[child_idx].is_exist()) {
			if (parent->child_nodes[child_idx]->is_expanded()
				&& parent->child_nodes[child_idx]->is_evaled())
			{
				print_ucb(nullptr, parent->child_nodes[child_idx].get());
				return parent->child_nodes[child_idx].get();
			}
		}
	}

	// parent から次に遷移出来なかった。
	return nullptr;
}



// print_ucb とか言いつつ、child_info とかを表示するだけ。
void print_ucb(uct_child_info_t* info, const UctNode* node) {
	const int& total = node->legal_moves_size;

	float reward_mean, bonus;
	float current_max_ucb_score = -1;    // スコアが負にならない前提。
	int current_max_idx = -1;

	// TODO: 他の変数についても
	// atomic からload
	const int node_move_count = node->move_count.load();
	const auto node_sum_value = node->sum_value.load();

	const auto this_puct_c_base = (info == nullptr ? g_puct_c_base_root : g_puct_c_base);
	const auto this_puct_c_init = (info == nullptr ? g_puct_c_init_root : g_puct_c_init);
	const float sqrt_sum = sqrtf(node_move_count);

	//const float C_s = logf((1.0 + node_move_count + this_puct_c_base) / this_puct_c_base) + this_puct_c_init;
	const float C_s = FastLog((1.0 + node_move_count + this_puct_c_base) / this_puct_c_base) + this_puct_c_init;

	// 未訪問node での値
	//const float init_reward_mean = (info != nullptr ? info->get_winrate() : 0.5);
	const float fpu_reduction = (
		info == nullptr ? g_puct_c_fpu_reduction_root : g_puct_c_fpu_reduction
		) * sqrtf(node->sum_visited_probability);
	const float init_reward_mean = (node_sum_value > 0 ?
		std::max(0.0f, (float)(node_sum_value / node_move_count) - fpu_reduction) : 0.0f);
	const float init_bonus = (node_move_count == 0 ? 1.0 : sqrt_sum);

	int n_child_win_moves = 0;    // 子が勝つ指し手の数(親が負ける指し手の数)
	for (int i = 0; i < total; ++i) {
		const auto& crrt_child_info = node->child_infos[i];
		if (crrt_child_info.is_win()) {    // 親の負け
			++n_child_win_moves;
			continue;
		}
		else if (crrt_child_info.is_lose()) {    // 親の勝ち
			if (info != nullptr) {
				//info->set_win();    // OrNode
			}
			//return i;
			continue;
		}

		const auto node_child_move_count = crrt_child_info.move_count.load();
		const auto node_child_sum_value = crrt_child_info.sum_value.load();

		if (node_child_move_count == 0) {
			// TODO: 親ノードの平均報酬で近似する。
			//       -> https://tadaoyamaoka.hatenablog.com/entry/2020/01/13/121234
			//          dlshogi では0 より0.5 の方が強かったようで、試す価値はあるかもしれん。
			//       -> いや、あれだな、向こうはpolicy の精度が良いからそうなるんじゃないかね。
			reward_mean = init_reward_mean;    // 弱くなった。
			//reward_mean = g_init_value;
			bonus = init_bonus;
		}
		else {
			reward_mean = node_child_sum_value / node_child_move_count;
			bonus = sqrt_sum / (1 + node_child_move_count);
		}
		const float ucb_score = reward_mean + C_s * bonus * node->child_infos[i].probability;

		if (ucb_score > current_max_ucb_score) {
			current_max_ucb_score = ucb_score;
			current_max_idx = i;
		}

		sync_cout
			<< "[" << i << "] " << crrt_child_info.move.toUSI() << "[" << crrt_child_info.move.value() << "]"
			<< ",move_cout = " << crrt_child_info.move_count
			<< ",probability = " << crrt_child_info.probability
			<< ",winrate = " << crrt_child_info.get_winrate()
			<< ",reward_mean = " << reward_mean
			<< ",bonus = " << bonus
			<< ",ucb_score = " << ucb_score
			<< sync_endl;
	}

	sync_cout << "info is_evaled = " << bts(node->is_evaled()) << sync_endl;

#if defined(DEBGU_PUCT_20231125_3)
	std::cout
		<< "[logf>apporx=" << n_logf_is_greater_than_approx
		<< "[logf>apporx av=" << logf_is_greater_than_approx_loged_av
		<< ",approx>logf=" << n_approx_is_greater_than_logf
		<< ",approx>logf av=" << approx_is_greater_than_logf_loged_av
		<< ",total" << n_select_max_ucb_child_called << "]" << std::endl;
#endif

	//if (n_child_win_moves == total) {
	//	if (info != nullptr) {
	//		//info->set_lose();    // AndNode
	//	}
	//}

	//if (info == nullptr && n_child_win_moves > 0) {
	//	sync_cout << "[" << n_child_win_moves << "]" << IO_UNLOCK;
	//}

	// TODO
	//     : このrandom、要らんから消そう
	//       -> それと同時に、vector にためる必要もないから、逐次最大値をとるインデックスを更新する方式にしよう。
	// この、初動の全くデータがなくてrandom に選ぶ時の奴、
	// 指し手のオーダリングとかしたらちょっと良くなりそう。
	// (スコアが同じ奴らの、オーダリングにおけるスコアを計算して、一番高い奴を選ぶ。)
	//return current_max_idx;
}


#endif    // PARALLEL_PUCT_PLAYER