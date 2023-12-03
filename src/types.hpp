#pragma once 
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>

//#include "cshogi.h"
#include "position.hpp"
#include "move.hpp"
#include "config.hpp"

// ==================================================
// cppshogi/cppshogi.h より
// ==================================================
struct HuffmanCodedPosAndEval2 {
	HuffmanCodedPos hcp;
	s16 eval;
	u16 bestMove16;
	uint8_t result; // xxxxxx11 : 勝敗、xxxxx1xx : 千日手、xxxx1xxx : 入玉宣言、xxx1xxxx : 最大手数
};
static_assert(sizeof(HuffmanCodedPosAndEval2) == 38, "");

struct HuffmanCodedPosAndEval3 {
	HuffmanCodedPos hcp; // 開始局面
	u16 moveNum; // 手数
	u8 result; // xxxxxx11 : 勝敗、xxxxx1xx : 千日手、xxxx1xxx : 入玉宣言、xxx1xxxx : 最大手数
	u8 opponent; // 対戦相手（0:自己対局、1:先手usi、2:後手usi）
};
static_assert(sizeof(HuffmanCodedPosAndEval3) == 36, "");

struct MoveInfo {
	u16 selectedMove16; // 指し手
	s16 eval; // 評価値
	u16 candidateNum; // 候補手の数
};
static_assert(sizeof(MoveInfo) == 6, "");

struct MoveVisits {
	MoveVisits() {}
	MoveVisits(const u16& move16, const u16& visitNum) : move16(move16), visitNum(visitNum) {}
	MoveVisits(const MoveVisits& obj) : move16(obj.move16), visitNum(obj.visitNum) {}
	MoveVisits(MoveVisits&& obj) noexcept : move16(std::move(obj.move16)), visitNum(std::move(obj.visitNum)) {}

	MoveVisits& operator=(const MoveVisits& obj) {
		move16 = obj.move16;
		visitNum = obj.visitNum;
		return *this;
	}

	MoveVisits& operator=(MoveVisits&& obj) noexcept {
		move16 = std::move(obj.move16);
		visitNum = std::move(obj.visitNum);
		return *this;
	}

	u16 move16;
	u16 visitNum;
};
static_assert(sizeof(MoveVisits) == 4, "");

struct Hcpe3CacheBody {
	HuffmanCodedPos hcp; // 局面
	int eval;
	float result;
	int count; // 重複カウント
};

struct Hcpe3CacheCandidate {
	u16 move16;
	float prob;
};

struct TrainingData {
	TrainingData(const HuffmanCodedPos& hcp, const int& eval, const float& result, const int& reserve_size = 32)
		: hcp(hcp), eval(eval), result(result), count(1)
	{
		candidates.reserve(reserve_size);
	};

	TrainingData(HuffmanCodedPos&& _hcp, const int& _eval, const float& _result, const int& _reserve_size = 32)
		: eval(_eval), result(_result), count(1)
	{
		hcp = std::move(_hcp);
		candidates.reserve(_reserve_size);
	};

	TrainingData(const Hcpe3CacheBody& body, const Hcpe3CacheCandidate* candidates, const size_t candidateNum)
		: hcp(body.hcp), eval(body.eval), result(body.result), count(body.count), candidates(candidateNum) {
		for (size_t i = 0; i < candidateNum; i++) {
			this->candidates.emplace(candidates[i].move16, candidates[i].prob);
		}
	};

	HuffmanCodedPos hcp;
	int eval;                                     // 200とか、3000とか、-1500 みたいな所謂評価値
	// 自分から見て有利なら+, 不利なら-
	float result;                                 // 手番側からみて勝ちなら1, 引き分けなら0.5, 負けなら0
	std::unordered_map<u16, float> candidates;    // NOTE
	//     : probability が入るっぽい。
	//     ; hcpe のような最善手のみが登録された教師では、
	//       candidates[最善手] = 1; という使われ方をしてるが、
	//       この1 も、この指し手の確率 = 100% と捉えることが出来る。
	int count; // 重複カウント
};

constexpr u8 GAMERESULT_SENNICHITE = 0x4;
constexpr u8 GAMERESULT_NYUGYOKU = 0x8;
constexpr u8 GAMERESULT_MAXMOVE = 0x16;

// ==================================================
// 自分で定義
// ==================================================

typedef struct Teacher {
	std::string sfen;
	int move;        // __move_to_usi() を使うと、usi形式に直せるよ。
	int ply;        // sfen に含まれているけど、別で格納。
	float value;    // TODO: これの仕様決める
	float result;    // 手番側から見て、0なら負け、1なら勝ち、0.5なら引き分け

	// https://stackoverflow.com/questions/13812703/c11-emplace-back-on-vectorstruct
	Teacher()
		: sfen(""), move(0), ply(0), value(0), result(0)
	{}

	// https://stackoverflow.com/questions/13812703/c11-emplace-back-on-vectorstruct
	Teacher(
		const std::string& arg_sfen, int arg_move, int arg_ply, float arg_value, float arg_result
	) : sfen(arg_sfen), move(arg_move), ply(arg_ply), value(arg_value), result(arg_result)
	{}

	Teacher(
		std::string&& arg_sfen, int arg_move, int arg_ply, float arg_value, float arg_result
    ) : sfen(std::move(arg_sfen)), move(arg_move), ply(arg_ply), value(arg_value), result(arg_result)
	{}
}teacher_t;

// HACK: YoBokkMoveInfo とかの方がしっくりくるかも。
// https://github.com/yaneurao/YaneuraOu/wiki/%E5%AE%9A%E8%B7%A1%E3%81%AE%E4%BD%9C%E6%88%90
// 一つの候補手についての情報
typedef struct YoBookMove {
	Move move1;    // self move
	Move move2;    // pondermove
	int eval;
	int depth;
	int select;

	YoBookMove(Move move1_, Move move2_, int eval_, int depth_, int select_)
		: move1(move1_), move2(move2_), eval(eval_), depth(depth_), select(select_)
	{

	}

	void print_usi() const {
		const auto& move1_usi = move1.toUSI();
		const auto& move2_usi = move2.toUSI();
		std::cout
			<< "info string"
			<< " depth " << depth << " score cp " << eval << " pv " << move1_usi << " " << move2_usi << std::endl;
	}

	static YoBookMove moveNone() { return YoBookMove(Move::moveNone(), Move::moveNone(), INT_MIN, INT_MIN, INT_MIN); }
} yo_book_move_t;

// map のkey/value で言うとvalue に相当する情報
// 候補手一覧を保持。
typedef struct YoBookValue {
	int ply;
	std::vector<yo_book_move_t> moves;     // 指し手リスト

	YoBookValue()
		: ply(-1)
	{

	}

	YoBookValue(const int& ply_)
		: ply(ply_)
	{

	}

	// HACK: get_best_eval_move() とかの方がしっくりくる。
	// 評価値を基準に最善手を返す。
	[[deprecated("Use get_best_eval_move() instead.")]]
	yo_book_move_t get_best_by_eval() const {
		return *std::max_element(moves.begin(), moves.end(), [](const auto& x, const auto& y) {
			return x.eval < y.eval;
		});
	}

	// 評価値が一番良く、更にその中でdepth が一番高い指し手(:=最善手)がeva_limit, depth_limit 共に満たす場合のみそれを返す。
	// やねうら王形式ならsort されてるはずだけど、念の為全探索する(depth 順にはsort されてなかったと思うので全探索して損無いはず。。。)。
	// @arg eval_limit
	//     : 最善手がこれ以上の評価値 であれば定跡に登録されてる最善手を返す。
	//     : 当然だが、自分目線の評価値である(自分が有利なら+, 不利なら-)。
	// @arg depth_limit
	//     : 最善手がこれ以上のdepth であれば定跡に登録されている最善手を返す。
	// @arg check_depth_limit_last
	//     : 一番最後にdepth_limit を確認する。
	//           : true の時
	//             1. eval_limit を満たし且つ評価値最大の中で、特にdepth 最大の指し手を見つける。
	//             2. その指し手のdepth がdepth_limit を満たすか確認する。
	//             3. 以上の条件に合致するものがあればそれを最善手として返す
	//             (一番最後に一度だけチェック)
	//           : false の時
	//             1. eval_limit とdepth_limit を満たし且つ評価値最大の中で、特にdepth 最大の指し手を見つける。
	//             2. 以上の条件に合致するものがあればそれを最善手として返す
	//             (走査するタイミングで任意の指し手においてチェック)
	//             (恐らくやねうら王はコッチ)
	// @return
	//     : first
	//           : 条件に合う指し手があったか否か。
	//             あればtrue で、second の指し手を指せばよい。
	//     : second
	//           : first==true なら条件に合った指し手。
	//             first==false ならYoBookMove::moveNone()。
	std::pair<bool, yo_book_move_t> get_best_eval_move(int eval_limit = INT_MIN, int depth_limit = 0, bool check_depth_limit_last = true) const {
		static constexpr int INIT_BESTMOVE_IDX = -1;
		int bestmove_idx = INIT_BESTMOVE_IDX;
		int bestmove_eval = INT_MIN;
		int bestmove_depth = INT_MIN;

		// eval 最大の手の中で、更にdepth 最大の指し手を探索。
		const int moves_size = moves.size();
		for (int i = 0; i < moves_size; ++i) {
			const auto& book_move = moves[i];
			if (book_move.eval < eval_limit) {    // 評価値が下限よりも悪い(小さい)
				continue;
			}
			if (!check_depth_limit_last) {
				if (book_move.depth < depth_limit) {    // 最善手のdepth が下限よりも浅い(小さい)
					continue;
				}
			}

			if (book_move.eval > bestmove_eval) {
				bestmove_idx = i;
				bestmove_eval = book_move.eval;
				bestmove_depth = book_move.depth;
			}
			else if (book_move.eval == bestmove_eval && book_move.depth > bestmove_depth) {
				bestmove_idx = i;
				bestmove_eval = book_move.eval;
				bestmove_depth = book_move.depth;
			}
		}

		// 基準以上の評価値 の指し手がそもそも存在しなかった || 最善手のdepth が下限よりも浅い(小さい)
		if (bestmove_idx == INIT_BESTMOVE_IDX) {
			return { false, YoBookMove::moveNone() };
		}
		else if (check_depth_limit_last && moves[bestmove_idx].depth < depth_limit) {
			return { false, YoBookMove::moveNone() };
		}
		else {
			return { true, moves[bestmove_idx] };
		}
	}

	void print_all_moves() const {
		const int& sum_select = std::accumulate(moves.begin(), moves.end(), 0, [](const int& acc, const yo_book_move_t& book_move) {
			return acc + book_move.select;
			});

		// debug
		std::cout << "info string sum_select=" << sum_select << std::endl;

		// やねうら王で確かsort されているはずで、こちらではsort しない。
		const int moves_size = moves.size();
		for (int i = 0; i < moves_size; ++i) {
			const auto& book_move = moves[i];
			std::cout
				<< "info string"
				<< " multipv " << i + 1
				<< " depth " << book_move.depth
				<< " score cp " << book_move.eval
				<< " pv " << book_move.move1.toUSI() << " " << book_move.move2.toUSI()
				<< " (" << std::setprecision(4) << (book_move.select * 100.0) / sum_select << "%)"
				<< std::endl;
		}
	}

} yo_book_value_t;

// HACK: これ、何故に継承にした？普通にメンバに持っちゃダメやったんか？ちょいきもないか？
// やねうら王形式の定跡が、一つの局面に対して持っている情報
typedef struct YoBook : public YoBookValue {
	std::string sfen;

	YoBook()
		: YoBookValue(), sfen("")
	{

	}

	YoBook(const std::string& sfen_, const int& ply_)
		: YoBookValue(ply_), sfen(sfen_)
	{

	}
} yo_book_t;

// go コマンドのparse の結果を保持
// http://shogidokoro.starfree.jp/usi.html
typedef struct GoInfo {
private:

public:
	static constexpr int TIME_NONE = -1;

	// TODO: こいつら、private に移してgetter, setter からのアクセスにしたい。
	bool is_infinite;
	bool is_ponder;    // この場合も、go inf と同様で、無限と仮定して思考する。ただ、GUI からはbtime, wtime 等が与えられる模様。
	int time[ColorNum];
	int inc[ColorNum];
	int byoyomi;

	// https://qiita.com/Nabetani/items/1f41e38cb92654ede6d8
	GoInfo()
	{
		reset();
	}


	std::string to_str() const {
		std::ostringstream oss;
		oss
			<< "GoInfo{"
			<< ", time[Black]=" << time[Black]
			<< ", time[White]=" << time[White]
			<< ", inc[Black]="  << inc[Black]
			<< ", inc[White]="  << inc[White]
			<< ", byoyomi="     << byoyomi
			<< ", is_infinite=" << std::boolalpha << is_infinite
			<< ", is_ponder="   << std::boolalpha << is_ponder
			<< "}";
		return oss.str();
	}
	
	// parse する前に毎度呼ぶべし
	void reset() {
		is_infinite = false;
		is_ponder = false;

        // 時間の初期値は0 とする。
		std::fill_n(time, ColorNum, 0);
		std::fill_n(inc, ColorNum, 0);
		byoyomi = 0;
	}

	// --------------------------------------------------
	// 以下は全て、完全にparse が完了してから呼び出すこと
	// --------------------------------------------------

	inline bool is_inc() const {
		return (inc[Black] > 0 && inc[White] > 0);
	}

	inline bool is_byoyomi() const {
		return (byoyomi > 0);
	}

	// HACK: もう完全に持ち時間が無い って取り違えちゃう名前な気がする。
	//       is_additional_time_zero とかに変えよう(これもなんかちゃう。なんかええ名前考えようお兄さん。)。
	// (http://shogidokoro.starfree.jp/usi.html
	// 将棋所は"秒読みと加算が両方とも0ならbyoyomi 0を送ります。")
	// @return: 切れ負け将棋であるか否か
	inline bool is_extra_time_zero(const Color& c) const {
		return (inc[c] == 0 && byoyomi == 0);
	}

	inline int get_time(const Color& c) const { return time[c]; }
	inline int get_inc(const Color& c) const { return inc[c]; }

	// 格納した結果に矛盾が無いかを確認する。
	// true を返すことが、矛盾が無いことの必要条件。
	bool check_state() const {
		// 加算 と 秒読み は互いに排他的なオプションであり、共存してはならない
		// (go inf の場合も、go inf と加算 又は秒読み は排他的 なので不適。)
		if (is_inc() && is_byoyomi()) {
			return false;
		}

		// 思考時間制限あり と思考時間無限 は互いに排他的なオプションであり、共存してはならない。
		// go infinite なら、少なくとも初期値0 から変化していないはず(0 がset された可能性は勿論あるが考えない)。
		if (is_infinite){
			if (!(time[Black] == 0 && time[White] == 0 && inc[Black] == 0 && inc[White] == 0 && byoyomi == 0)) {
				return false;
			}
		}
		return true;
	}
}go_info_t;

typedef struct ReadingHecpe3Task {
	ReadingHecpe3Task(
		const HuffmanCodedPosAndEval3& _hcpe3,
		const std::vector<std::vector<MoveVisits>>& _candidates_list,
		const std::vector<MoveInfo>& _move_info_list
	)
		: hcpe3(_hcpe3), candidates_list(_candidates_list), move_info_list(_move_info_list)
	{

	}

	ReadingHecpe3Task()
	{

	}


	HuffmanCodedPosAndEval3 hcpe3;    // 開始局面
	std::vector<std::vector<MoveVisits>> candidates_list;    // 一局分の情報が入る。
	std::vector<MoveInfo> move_info_list;                    // 一局分の情報が入る
}reading_hcpe3_task_t;

// HACK
//     : _t に。
//     : これ、pair のところ、流石にstruct 定義しても良かったんやないか？こりゃ分かりづらいよ。。。
// puct player で使用。
#if defined(FEATURE_V2)
    #if defined(USE_PACKED_FEATURE)
        using feature_t = std::tuple<char*, char*>;
		using unpacked_feature_t = std::tuple<float*, float*>;    // nn_tensorrt.hpp で使用。
#else
        using feature_t = std::tuple<float*, float*>;
    #endif
#else
    using feature_t = float*;
#endif