#ifndef CSHOGI_H
#define CSHOGI_H

#include <numeric>

#include "init.hpp"
#include "position.hpp"
#include "generateMoves.hpp"
#include "usi.hpp"
#include "book.hpp"
#include "square.hpp"
#include "mate.h"

#include "config.hpp"
#if not defined(FOR_BUILD_CSHOGI)
#include "types.hpp"
#endif
#if defined(DFPN_CSHOGI)
#include "dfpn.h"
#elif defined(DFPN_DLSHOGI)
#include "dfpn_dlshogi.hpp"
#elif defined(DFPN_PARALLEL)
#include "my_dfpn_parallel.hpp"
#elif defined(DFPN_PARALLEL2)
#include "my_dfpn_parallel2.hpp"
#elif defined(DFPN_PARALLEL3)
#include "my_dfpn_parallel3.hpp"
#endif

//template <typename T> constexpr bool false_v = false;

// ==================================================
// 入力特徴量
// ==================================================
////////// 入力特徴量のための定数(共通)
constexpr int PIECETYPE_NUM = 14; // 駒の種類

////////// 自分の入力特徴量関連
#ifdef FEATURE_V0
//// 説明
// 構造
//     : feature1_black, feature2_black, feature1_white, feature2_white の順で、
//       idx が小さい方から順にチャンネルが並んでる。
// 各特徴
//     : feature1
//       各駒の位置
//     : feature2
//       各持ち駒の数
//       ある持ち駒のfeature2 のチャンネルの先頭のインデックスをhp_idx とし、持ち駒がnum 枚あるとすると、
//       hp_idx からnum 枚のチャンネル全てにフラグを立てる。
//       但し、num が上限値N_MAX_HXXXX を超える場合はその上限値でclip する。
//// 盤上の駒についての入力特徴量について
constexpr u32 N_FEATURE1_CHANNEL_PER_COLOR = PIECETYPE_NUM;

//// 持ち駒についての入力特徴量について
// 各持ち駒の、入力特徴量の手番ごとの(チャンネル)数
// HACK: config.hpp を読み込んじゃうと、cshogi_test0_cy でも関係ないconfig.hpp が必要に。。。
#ifndef PYTHON_DLSHOGI2
// 私の現状の値
constexpr int N_MAX_HPAWN = 8; // 歩の持ち駒の上限
#else
// python-dlshogi2対応
constexpr int N_MAX_HPAWN = 18; // 歩の持ち駒の上限
#endif
constexpr int N_MAX_HLANCE = 4;
constexpr int N_MAX_HKNIGHT = 4;
constexpr int N_MAX_HSILVER = 4;
constexpr int N_MAX_HGOLD = 4;
constexpr int N_MAX_HBISHOP = 2;
constexpr int N_MAX_HROOK = 2;
constexpr u32 N_MAX_HANDS[] = {
	N_MAX_HPAWN, // PAWN
	N_MAX_HLANCE, // LANCE
	N_MAX_HKNIGHT, // KNIGHT
	N_MAX_HSILVER, // SILVER
	N_MAX_HGOLD, // GOLD
	N_MAX_HBISHOP, // BISHOP
	N_MAX_HROOK // ROOK
};    // HandPiece と同じ並び。
// 入力特徴量とする、持ち駒の総数
constexpr u32 N_FEATURE2_CHANNEL_PER_COLOR = N_MAX_HPAWN + N_MAX_HLANCE + N_MAX_HKNIGHT + N_MAX_HSILVER + N_MAX_HGOLD + N_MAX_HBISHOP + N_MAX_HROOK;

//// 全ての入力特徴量について
// 総入力特徴量数(片方の手番だけのチャンネル数。本当の総数は、N_FEATURE_CHANNEL = 2 * N_FEATURE_CHANNEL_PER_COLOR)
// HACK: 2 * ってしてるけど、本当は、ColorNum * の方が良い。
constexpr u32 N_FEATURE_WIDTH = 9;      // 画像の横幅
constexpr u32 N_FEATURE_HEIGHT = 9;     // 画像の縦幅
constexpr u32 N_FEATURE_CHANNEL_PER_COLOR = N_FEATURE1_CHANNEL_PER_COLOR + N_FEATURE2_CHANNEL_PER_COLOR;    // 入力特徴量の手番ごとのチャンネル枚数
constexpr u32 N_FEATURE_CHANNEL = 2 * N_FEATURE_CHANNEL_PER_COLOR;    // 入力特徴量の合計チャンネル枚数

constexpr u32 N_FEATURE_WH = N_FEATURE_WIDTH * N_FEATURE_HEIGHT;    // 入力特徴量の1枚(1チャンネル)ごとのサイズ
constexpr u32 N_FEATURE_WHC_PER_COLOR =
N_FEATURE_WIDTH * N_FEATURE_HEIGHT * N_FEATURE_CHANNEL_PER_COLOR;    // 入力特徴量の手番ごとの合計サイズ
constexpr u32 N_FEATURE_WHC = 2 * N_FEATURE_WHC_PER_COLOR;    // 入力特徴量の合計サイズ

//// 使用箇所について
// N_FEATURE_WHC
//     : nn_tensorrt_test.hpp
//           : cudaHostAlloc() にて
//     : make_input_features_test.hpp
//           : std::vector<float[N_FEATURE_WHC]> _eval_queue;
//     : puct_player.hpp
//           : cudaHostAlloc() にて
//           : _eval_queue にて、各batch が入力特徴量をstore するべき開始位置を計算する為に
//     : parallel_puct_player.hpp
//           : cudaHostAlloc() にて
//           : _eval_queue にて、各batch が入力特徴量をstore するべき開始位置を計算する為に
//     : nn_tensorrt.cpp
//           : cudaMalloc() にて
//           : 入力特徴量のcudaMemcpyAsync() にて
//     : dataloader.h
//           : _store_one_teacher<teacher_t>() にて、make_input_features() に渡す入力特徴量の開始位置の計算
//           : _store_one_teacher<TrainingData>() にて、make_input_features() に渡す入力特徴量の開始位置の計算
//     : cshogi.h
// N_FEATURE_CHANNEL
//     : make_input_features_test.hpp
// N_FEATURE_WH
// N_FEATURE2_CHANNEL_PER_COLOR
// N_FEATURE1_CHANNEL_PER_COLOR
// N_FEATURE_WHC_PER_COLOR
// 
//// cython 用
constexpr u32 _N_FEATURE_WIDTH = N_FEATURE_WIDTH;
constexpr u32 _N_FEATURE_HEIGHT = N_FEATURE_HEIGHT;
constexpr u32 _N_FEATURE_CHANNEL = N_FEATURE_CHANNEL;

constexpr u32 _N_FEATURE_WH = N_FEATURE_WH;
constexpr u32 _N_FEATURE_WHC = N_FEATURE_WHC;
#elif defined(FEATURE_V1)
//// 説明
// 構造
//     : feature1_black, feature1_white, feature2_black, feature2_white, feature3 の順で、
//       idx が小さい方から順にチャンネルが並んでる。
//     : black, white とは便宜上の手番であり、現在の手番側を便宜上black として扱う。
//       (手番側の情報が常に各特徴(1, 2)の前半のチャンネルに来るようにする)
//     : 当然盤面も、現在の手番側を先手と考えて見るので、適宜回転している。
// 各特徴
//     : feature1
//       盤上の各駒の位置
//     : feature2
//       各持ち駒の数
//       ある持ち駒のfeature2 のチャンネルの先頭のインデックスをhp_idx とし、持ち駒がnum 枚あるとすると、
//       hp_idx からnum 枚のチャンネル全てにフラグを立てる。
//       但し、num が上限値N_MAX_HXXXX を超える場合はその上限値でclip する。
//     : feature3
//       王手されてたら1
//// feature1
constexpr u32 N_FEATURE1_CHANNEL_PER_COLOR = PIECETYPE_NUM;
constexpr u32 N_FEATURE1_CHANNEL = (u32)ColorNum * N_FEATURE1_CHANNEL_PER_COLOR;

//// feature2
// 各持ち駒の、入力特徴量の手番ごとの(チャンネル)数
// HACK: config.hpp を読み込んじゃうと、cshogi_test0_cy でも関係ないconfig.hpp が必要に。。。
constexpr int N_MAX_HPAWN = 8; // 歩の持ち駒の上限
constexpr int N_MAX_HLANCE = 4;
constexpr int N_MAX_HKNIGHT = 4;
constexpr int N_MAX_HSILVER = 4;
constexpr int N_MAX_HGOLD = 4;
constexpr int N_MAX_HBISHOP = 2;
constexpr int N_MAX_HROOK = 2;
constexpr u32 N_MAX_HANDS[] = {
	N_MAX_HPAWN, // PAWN
	N_MAX_HLANCE, // LANCE
	N_MAX_HKNIGHT, // KNIGHT
	N_MAX_HSILVER, // SILVER
	N_MAX_HGOLD, // GOLD
	N_MAX_HBISHOP, // BISHOP
	N_MAX_HROOK // ROOK
};    // HandPiece と同じ並び。
// 入力特徴量とする、持ち駒の総数
constexpr u32 N_FEATURE2_CHANNEL_PER_COLOR = N_MAX_HPAWN + N_MAX_HLANCE + N_MAX_HKNIGHT + N_MAX_HSILVER + N_MAX_HGOLD + N_MAX_HBISHOP + N_MAX_HROOK;
constexpr u32 N_FEATURE2_CHANNEL = (u32)ColorNum * N_FEATURE2_CHANNEL_PER_COLOR;

//// feature3
constexpr u32 N_FEATURE3_CHANNEL = 1;

//// 全ての入力特徴量について
// 総入力特徴量数(片方の手番だけのチャンネル数。本当の総数は、N_FEATURE_CHANNEL = 2 * N_FEATURE_CHANNEL_PER_COLOR)
// HACK: 2 * ってしてるけど、本当は、ColorNum * の方が良い。
constexpr u32 N_FEATURE_WIDTH = 9;      // 画像の横幅
constexpr u32 N_FEATURE_HEIGHT = 9;     // 画像の縦幅
constexpr u32 N_FEATURE_CHANNEL = N_FEATURE1_CHANNEL + N_FEATURE2_CHANNEL + N_FEATURE3_CHANNEL;    // 入力特徴量の合計チャンネル枚数

constexpr u32 N_FEATURE_WH = N_FEATURE_WIDTH * N_FEATURE_HEIGHT;    // 入力特徴量の1枚(1チャンネル)ごとのサイズ
constexpr u32 N_FEATURE_WHC = N_FEATURE_WH * N_FEATURE_CHANNEL;    // 入力特徴量の合計サイズ

//// cython 用
constexpr u32 _N_FEATURE_CHANNEL = N_FEATURE_CHANNEL;
constexpr u32 _N_FEATURE_WIDTH = N_FEATURE_WIDTH;
constexpr u32 _N_FEATURE_HEIGHT = N_FEATURE_HEIGHT;

constexpr u32 _N_FEATURE_WH = N_FEATURE_WH;
constexpr u32 _N_FEATURE_WHC = N_FEATURE_WHC;


#elif defined(FEATURE_V2)
////////// 入力特徴量のための定数(dlshogi互換)
//// 説明
// 構造
//     : features1, features2 の2つのTensor を入力とする。
//     : feature1
//           : 駒の配置_black, 駒の利き_black, 利きの数_black, 駒の配置_white, 駒の利き_white, 利きの数_white
//             の順で構成されている。
//           : black とは、便宜上の先手を指す(つまり手番側である)。
//           : 駒の利き_black とは、手番側の各駒の利きである。
//             例えば歩のチャンネルなら、手番側の歩の利きがある箇所に1 が立つ。
//           : 利きの数_black とは、手番側がsq に利きを通している駒の数(つまり利きの数) であり、
//             0個-> 000, 1個-> 100, 2個-> 110, 3個-> 111 となっている。
//             (100 なら、先頭のチャンネルのみ81pixel 全て1 で、残りの2つのチャンネルは全て0 を指す)
//     : feature2
//           : 持ち駒の数_black, 持ち駒の数_white, 王手か否か
//           : 持ち駒の数, 王手か否かは共にFEATURE_V1 での仕様と同じもの。
//// feature1
constexpr int MAX_ATTACK_NUM = 3; // 利き数の最大値
constexpr u32 MAX_FEATURES1_NUM = PIECETYPE_NUM/*駒の配置*/ + PIECETYPE_NUM/*駒の利き*/ + MAX_ATTACK_NUM/*利き数*/;

constexpr u32 N_FEATURE1_CHANNEL_PER_COLOR = MAX_FEATURES1_NUM;
constexpr u32 N_FEATURE1_CHANNEL = (u32)ColorNum * N_FEATURE1_CHANNEL_PER_COLOR;

//// feature2
constexpr int MAX_HPAWN_NUM = 8; // 歩の持ち駒の上限
constexpr int MAX_HLANCE_NUM = 4;
constexpr int MAX_HKNIGHT_NUM = 4;
constexpr int MAX_HSILVER_NUM = 4;
constexpr int MAX_HGOLD_NUM = 4;
constexpr int MAX_HBISHOP_NUM = 2;
constexpr int MAX_HROOK_NUM = 2;
constexpr u32 MAX_PIECES_IN_HAND[] = {
	MAX_HPAWN_NUM, // PAWN
	MAX_HLANCE_NUM, // LANCE
	MAX_HKNIGHT_NUM, // KNIGHT
	MAX_HSILVER_NUM, // SILVER
	MAX_HGOLD_NUM, // GOLD
	MAX_HBISHOP_NUM, // BISHOP
	MAX_HROOK_NUM, // ROOK
};
constexpr u32 MAX_PIECES_IN_HAND_SUM = MAX_HPAWN_NUM + MAX_HLANCE_NUM + MAX_HKNIGHT_NUM + MAX_HSILVER_NUM + MAX_HGOLD_NUM + MAX_HBISHOP_NUM + MAX_HROOK_NUM;
// NOTE
//     : 基本的には、任意のコマについて、
//       持ち駒が1枚なら1channelが1埋めされ、持ち駒が2枚なら2channelが1埋めされる(任意のchannelは9*9のサイズを持つ)。
//       ただ、歩に関しては最大で18枚あるが、18枚も持つことはレアなので、8channel分しか持たず、
//       持ち駒の歩が8枚を超えても、8として入力特徴量が作成される。
//       又、channelは、持ち駒にある枚数に応じてindex が小さいものから埋まっていく。
//     : +1/*王手*/ とあるが、これは王手が掛かっているか否か。
constexpr u32 MAX_FEATURES2_HAND_NUM = (int)ColorNum * MAX_PIECES_IN_HAND_SUM;
constexpr u32 MAX_FEATURES2_NUM = MAX_FEATURES2_HAND_NUM + 1/*王手*/;

constexpr u32 N_FEATURE2_CHANNEL = MAX_FEATURES2_NUM;


//// 全ての入力特徴量について
constexpr u32 N_FEATURE_WIDTH = 9;      // 画像の横幅
constexpr u32 N_FEATURE_HEIGHT = 9;     // 画像の縦幅
constexpr u32 N_FEATURE_CHANNEL = N_FEATURE1_CHANNEL + N_FEATURE2_CHANNEL;

constexpr u32 N_FEATURE_WH = N_FEATURE_WIDTH * N_FEATURE_HEIGHT;
constexpr u32 N_FEATURE1_WHC = N_FEATURE_WIDTH * N_FEATURE_HEIGHT * (N_FEATURE1_CHANNEL);
constexpr u32 N_FEATURE2_WHC = N_FEATURE_WIDTH * N_FEATURE_HEIGHT * (N_FEATURE2_CHANNEL);
constexpr u32 N_FEATURE_WHC = N_FEATURE_WIDTH * N_FEATURE_HEIGHT * (N_FEATURE_CHANNEL);

//// packed feature
// feature1, feature2 を表現するのに必要なchar の数。(1byte 変数が最低何個あれば良いか。)
constexpr u32 N_PACKED_FEATURE1_CHAR = (N_FEATURE1_WHC + 7) / 8;
constexpr u32 N_PACKED_FEATURE2_CHAR = (N_FEATURE2_CHANNEL + 7) / 8;

#if defined(USE_PACKED_FEATURE)
// gpu に転送する特徴量の配列の要素数
constexpr u32 N_FEATURE1 = N_PACKED_FEATURE1_CHAR;
constexpr u32 N_FEATURE2 = N_PACKED_FEATURE2_CHAR;
// 確保すべき容量(byte)
constexpr u32 SIZEOF_FEATURE1 = sizeof(char) * N_PACKED_FEATURE1_CHAR;
constexpr u32 SIZEOF_FEATURE2 = sizeof(char) * N_PACKED_FEATURE2_CHAR;
#else
// gpu に転送する特徴量の配列の要素数
constexpr u32 N_FEATURE1 = N_FEATURE1_WHC;
constexpr u32 N_FEATURE2 = N_FEATURE2_WHC;
// 確保すべき容量(byte)
constexpr u32 SIZEOF_FEATURE1 = sizeof(float) * N_FEATURE1_WHC;
constexpr u32 SIZEOF_FEATURE2 = sizeof(float) * N_FEATURE2_WHC;
#endif

//// cython 用
// TODO
//     : 以下の実装
//     : cython で対応
//     : fast_dataloader(cpp) で対応
//     : fast_dataloader(cython) で対応
//     : python で対応
//           : fast_dataloader, cshogi でfeature のversion の一致を確認。
//     : 推論部で対応
//           : 対応が必要なのは、puct_player, nn の二つかな？
constexpr u32 _N_FEATURE1_CHANNEL = N_FEATURE1_CHANNEL;
constexpr u32 _N_FEATURE1_WIDTH = N_FEATURE_WIDTH;
constexpr u32 _N_FEATURE1_HEIGHT = N_FEATURE_HEIGHT;

constexpr u32 _N_FEATURE2_CHANNEL = N_FEATURE2_CHANNEL;
constexpr u32 _N_FEATURE2_WIDTH = N_FEATURE_WIDTH;
constexpr u32 _N_FEATURE2_HEIGHT = N_FEATURE_HEIGHT;

constexpr u32 _N_FEATURE_CHANNEL = N_FEATURE_CHANNEL;
constexpr u32 _N_FEATURE_WIDTH = N_FEATURE_WIDTH;
constexpr u32 _N_FEATURE_HEIGHT = N_FEATURE_HEIGHT;
constexpr u32 _N_FEATURE_WH = N_FEATURE_WH;
constexpr u32 _N_FEATURE_WHC = N_FEATURE_WHC;

#endif

inline int _feature_ver() {
	return feature_ver();
}


// ==================================================
////////// 移動関連の定数
//// 自分の定数 (NNの出力関連)
// NOTE
//     : NNの着手のpolicyの出力を処理する際の定数なので、label を用いて命名している。
//     : 実際はNNの出力は1次元だが、便宜上画像のように考えて命名している。
// 各移動の方向(種類,channel)が、移動先として選べる升の数。
// 例えば歩は、敵陣地の一番奥には着手出来ないが、処理が面倒なので一律で81とする。
constexpr u32 N_LABEL_SQ_PER_CHANNEL = 9 * 9;

// NOTE
//     : MOVE_DIRECTION_NUM == N_LABEL_WITHOUT_PRO + N_LABEL_WITH_PRO
// 各移動の種類のチャンネル数。
constexpr u32 N_LABEL_WITHOUT_PRO = 10;       // 8方向 + 桂馬2方向 = 計10個。不成。
constexpr u32 N_LABEL_WITH_PRO = 10;          // 移動して且つ成る。
constexpr u32 N_LABEL_DROP = HandPieceNum;    // 持ち駒を打つ
constexpr u32 N_LABEL_CHANNEL
= N_LABEL_WITHOUT_PRO + N_LABEL_WITH_PRO + N_LABEL_DROP;    // 着手の種類の総数

// ラベルの合計サイズ。(27 * 81 = 2187)
constexpr u32 N_LABEL_SIZE = N_LABEL_CHANNEL * N_LABEL_SQ_PER_CHANNEL;

//// cython用
constexpr u32 _N_LABEL_SQ_PER_CHANNEL = N_LABEL_SQ_PER_CHANNEL;
constexpr u32 _N_LABEL_CHANNEL = N_LABEL_CHANNEL;
constexpr u32 _N_LABEL_SIZE = N_LABEL_SIZE;

//// cshogi由来の定数
// NOTE
//     : あくまで移動方向であることに注意。
//       UP であることは分かっても、UP に何マス進んだかは分からない。
// 移動方向の定数
enum MOVE_DIRECTION {
	UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT,
	UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE,
	MOVE_DIRECTION_NUM
};

// ==================================================
//inline int __dlshogi_get_features1_num() {
//	return 2 * MAX_FEATURES1_NUM;
//}
//inline int __dlshogi_get_features2_num() {
//	return MAX_FEATURES2_NUM;
//}


inline bool nyugyoku(const Position& pos);

inline void HuffmanCodedPos_init() {
	HuffmanCodedPos::init();
}

inline void PackedSfen_init() {
	PackedSfen::init();
}

inline void Book_init() {
	Book::init();
}

inline void Effect8_init() {
	Effect8::init();
}

inline std::string __to_usi(const int move) {
	return Move(move).toUSI();
}

inline std::string __to_csa(const int move) {
	return Move(move).toCSA();
}

unsigned short __move16_from_psv(const unsigned short move16);

class __Board
{
public:
	__Board() : pos(DefaultStartPositionSFEN) {}
	__Board(const std::string& sfen) : pos(sfen) {}
	~__Board() {}

	void set(const std::string& sfen) {
		history.clear();
		pos.set(sfen);
	}

	// position をset し、position に含まれる要素(初期局面 & 棋譜 & 初期局面のhash key)をそれぞれ抽出してポインタの指すアドレスに代入。
	// @arg position
	//     : startpos moves xxxx xxxx xxxx ...
	//     : [sfen から始まる局面] moves xxxx xxxx xxxx ...
	// @arg startsfen
	//     : nullptr 出ない時、parseした結果得られた初期局面がsfen形式で代入される。
	// @return : 成功したか否か
	bool _set_position_and_get_components(
		const std::string& position, std::string* startsfen, std::vector<Move>* moves, Key* startpos_key
	) {
		history.clear();
		std::istringstream ssPosCmd(position);
		std::string token;
		std::string sfen;

		ssPosCmd >> token;

		if (token == "startpos") {
			sfen = DefaultStartPositionSFEN;
			ssPosCmd >> token; // "moves" が入力されるはず。
		}
		else if (token == "sfen") {
			while (ssPosCmd >> token && token != "moves")
				sfen += token + " ";
		}
		else
			return false;

		pos.set(sfen);
		if (startsfen != nullptr) {
			*startsfen = sfen;
		}
		if (startpos_key != nullptr) {
			*startpos_key = pos.getKey();
		}

		// NOTE: "moves" の次からのtoken を取得していく。
		while (ssPosCmd >> token) {
			const Move move = usiToMove(pos, token);
			if (moves != nullptr) {
				moves->emplace_back(move);
			}
			if (!move) return false;
			push(move.value());
		}

		return true;
	}


	bool set_position(const std::string& position) {
		return _set_position_and_get_components(position, nullptr, nullptr, nullptr);
	}

	bool set_position_and_get_components(
		const std::string& position, std::string* startsfen, std::vector<Move>* moves, Key* startpos_key
	) {
		return _set_position_and_get_components(position, startsfen, moves, startpos_key);
	}


	void set_pieces(const int pieces[], const int pieces_in_hand[][7]) {
		history.clear();
		pos.set((const Piece*)pieces, pieces_in_hand);
	}
	bool set_hcp(char* hcp) {
		history.clear();
		return pos.set_hcp(hcp);
	}
	// position にあった奴をラップ。
	bool set_hcp(const HuffmanCodedPos& hcp) {
		history.clear();
		return pos.set(hcp);
	}
	bool set_psfen(char* psfen) {
		history.clear();
		return pos.set_psfen(psfen);
	}

	void reset() {
		history.clear();
		pos.set(DefaultStartPositionSFEN);
	}

	std::string dump() const {
		std::stringstream ss;
		pos.print(ss);
		return ss.str();
	}

	void push(const int move) {
		history.emplace_back(move, StateInfo());
		pos.doMove(Move(move), history.back().second);
	}

	void push(const Move move) {
		history.emplace_back(move, StateInfo());
		pos.doMove(move, history.back().second);
	}

	int pop() {
		const auto& move = history.back().first;
		pos.undoMove(move);
		history.pop_back();
		return move.value();
	}

	int peek() {
		if (history.size() > 0)
			return history.back().first.value();
		else
			return Move::moveNone().value();
	}

	void push_pass() {
		history.emplace_back(Move::moveNull(), StateInfo());
		pos.doNullMove<true>(history.back().second);
	}
	void pop_pass() {
		pos.doNullMove<false>(history.back().second);
		history.pop_back();
	}

	std::vector<int> get_history() const {
		std::vector<int> result;
		result.reserve(history.size());
		for (auto& m : history) {
			result.emplace_back((int)m.first.value());
		}
		return std::move(result);
	}

	bool is_game_over() const {
		const MoveList<LegalAll> ml(pos);
		return ml.size() == 0;
	}

	RepetitionType isDraw(const int checkMaxPly) const { return pos.isDraw(checkMaxPly); }

	int move(const int from_square, const int to_square, const bool promotion) const {
		if (promotion)
			return makePromoteMove<Capture>(pieceToPieceType(pos.piece((Square)from_square)), (Square)from_square, (Square)to_square, pos).value();
		else
			return makeNonPromoteMove<Capture>(pieceToPieceType(pos.piece((Square)from_square)), (Square)from_square, (Square)to_square, pos).value();
	}

	int drop_move(const int to_square, const int drop_piece_type) const {
		return makeDropMove((PieceType)drop_piece_type, (Square)to_square).value();
	}

	Move Move_from_usi(const std::string& usi) const {
		return usiToMove(pos, usi);
	}

	int move_from_usi(const std::string& usi) const {
		return usiToMove(pos, usi).value();
	}

	int move_from_csa(const std::string& csa) const {
		return csaToMove(pos, csa).value();
	}

	int move_from_move16(const unsigned short move16) const {
		return move16toMove(Move(move16), pos).value();
	}

	int move_from_psv(const unsigned short move16) const {
		return move16toMove(Move(__move16_from_psv(move16)), pos).value();
	}

	Color turn() const { return pos.turn(); }
	void setTurn(const int turn) { pos.setTurn((Color)turn); }
	int ply() const { return pos.gamePly(); }
	void setPly(const int ply) { pos.setStartPosPly(ply); }
	std::string toSFEN() const { return pos.toSFEN(); }
	std::string toCSAPos() const { return pos.toCSAPos(); }
	void toHuffmanCodedPos(char* data) const { pos.toHuffmanCodedPos((u8*)data); }
	void toPackedSfen(char* data) const { pos.toPackedSfen((u8*)data); }
	int piece(const int sq) const { return (int)pos.piece((Square)sq); }
	// NOTE: 王手が掛かっているかいるか否か。
	bool inCheck() const { return pos.inCheck(); }
	int mateMoveIn1Ply() { return pos.mateMoveIn1Ply().value(); }
	// NOTE
	//     : 詰ませられない場合は、u32 MoveNone = 0; が返ってくる。
	//     : ply 手以内に詰ませられる場合は、この局面で詰ませるのに指すべき手を返す。
	int mateMove(int ply) {
		if (pos.inCheck())
			return mateMoveInOddPlyReturnMove<true>(pos, ply).value();
		else
			return mateMoveInOddPlyReturnMove<false>(pos, ply).value();
	}
	// NOTE: 手番側が詰んでいるか否か
	bool is_mate(int ply) {
		return mateMoveInEvenPly(pos, ply);
	}
	unsigned long long getKey() const { return pos.getKey(); }    // boardKey + handKey
	unsigned long long getBoardKey() const { return pos.getBoardKey(); }    // boardKey
	bool moveIsPseudoLegal(const int move) const { return pos.moveIsPseudoLegal(Move(move)); }
	bool moveIsLegal(const int move) const { return pos.moveIsLegal(Move(move)); }
	bool is_nyugyoku() const { return nyugyoku(pos); }
	bool isOK() const { return pos.isOK(); }

	std::vector<int> pieces_in_hand(const int color) const {
		const Hand h = pos.hand((Color)color);
		return std::vector<int>{
			(int)h.numOf<HPawn>(), (int)h.numOf<HLance>(), (int)h.numOf<HKnight>(), (int)h.numOf<HSilver>(), (int)h.numOf<HGold>(), (int)h.numOf<HBishop>(), (int)h.numOf<HRook>()
		};
	}

	std::vector<int> pieces() const {
		std::vector<int> board(81);

		bbToVector(Pawn, Black, BPawn, board);
		bbToVector(Lance, Black, BLance, board);
		bbToVector(Knight, Black, BKnight, board);
		bbToVector(Silver, Black, BSilver, board);
		bbToVector(Bishop, Black, BBishop, board);
		bbToVector(Rook, Black, BRook, board);
		bbToVector(Gold, Black, BGold, board);
		bbToVector(King, Black, BKing, board);
		bbToVector(ProPawn, Black, BProPawn, board);
		bbToVector(ProLance, Black, BProLance, board);
		bbToVector(ProKnight, Black, BProKnight, board);
		bbToVector(ProSilver, Black, BProSilver, board);
		bbToVector(Horse, Black, BHorse, board);
		bbToVector(Dragon, Black, BDragon, board);

		bbToVector(Pawn, White, WPawn, board);
		bbToVector(Lance, White, WLance, board);
		bbToVector(Knight, White, WKnight, board);
		bbToVector(Silver, White, WSilver, board);
		bbToVector(Bishop, White, WBishop, board);
		bbToVector(Rook, White, WRook, board);
		bbToVector(Gold, White, WGold, board);
		bbToVector(King, White, WKing, board);
		bbToVector(ProPawn, White, WProPawn, board);
		bbToVector(ProLance, White, WProLance, board);
		bbToVector(ProKnight, White, WProKnight, board);
		bbToVector(ProSilver, White, WProSilver, board);
		bbToVector(Horse, White, WHorse, board);
		bbToVector(Dragon, White, WDragon, board);

		return std::move(board);
	}

	void piece_planes(char* mem) const {
		// P1 piece 14 planes
		// P2 piece 14 planes
		float* data = (float*)mem;
		for (Color c = Black; c < ColorNum; ++c) {
			for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
				Bitboard bb = pos.bbOf(pt, c);
				while (bb) {
					const Square sq = bb.firstOneFromSQ11();
					data[sq] = 1.0f;
				}
				data += 81;
			}
		}
	}

	// 白の場合、盤を反転するバージョン
	void piece_planes_rotate(char* mem) const {
		// P1 piece 14 planes
		// P2 piece 14 planes
		if (pos.turn() == Black) {
			// 黒の場合
			piece_planes(mem);
			return;
		}
		// 白の場合
		float* data = (float*)mem;
		for (Color c = White; c >= Black; --c) {
			for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
				Bitboard bb = pos.bbOf(pt, c);
				while (bb) {
					// 盤面を180度回転
					const Square sq = SQ99 - bb.firstOneFromSQ11();
					data[sq] = 1.0f;
				}
				data += 81;
			}
		}
	}

#ifdef FEATURE_V0

	// 盤上の駒について、フラグを立てる。
	// 色がcのplyaer の駒pt がsq に存在するというフラグを立てる。
	static void set_feature(float* dst, const Color c, const PieceType pt, const Square sq) {
		// PieceType は、occupied == 0, Pawn == 1, Lnace == 2, ...となっている。
		// 今回はoccupied は考慮しないので、その分を引いて(pt -1)としている。
		const u32 idx = (u32)c * N_FEATURE_WHC_PER_COLOR + (u32)(pt - 1) * N_FEATURE_WH + (u32)sq;
		dst[idx] = 1;
	}

	// 持ち駒について、フラグを立てる。
	// num枚のチャンネル全てにフラグを立てる。(81 * num個のフラグを立てる。)
	// @arg c: 与えられた持ち駒hp を持っている方のplayerの色
	// @arg hp: 持ち駒の種類
	// @arg num: 持ち駒の数
	static void set_feature_hand(float* dst, const Color c, const HandPiece hp, const u32 num) {
		// 引数で渡されたhp の手前までの、各駒の持ち駒の上限数の和。
		// つまり、hp の持ち駒についてのチャンネルの先頭のchannel index.(チャンネル番号)
		const u32 tmp = std::accumulate(std::begin(N_MAX_HANDS), std::next(std::begin(N_MAX_HANDS), (u32)hp), 0);
		const u32 idx = (u32)c * N_FEATURE_WHC_PER_COLOR + (N_FEATURE1_CHANNEL_PER_COLOR + tmp) * N_FEATURE_WH;
		std::fill_n(dst + idx, N_FEATURE_WH * num, 1);
	}

	// NOTE
	//     : 因みに、出来上がった配列の、例えば初期局面の先手の歩についてを見てみると、
	//       0 0 0 0 0 0 1 0 0
	//       0 0 0 0 0 0 1 0 0
	//       0 0 0 0 0 0 1 0 0
	//       0 0 0 0 0 0 1 0 0
	//       0 0 0 0 0 0 1 0 0
	//       0 0 0 0 0 0 1 0 0
	//       0 0 0 0 0 0 1 0 0
	//       0 0 0 0 0 0 1 0 0
	//       0 0 0 0 0 0 1 0 0
	//      となっている。
	// @arg dst
	//     : 形式上は(N_FEATURE_CHANNEL, N_FEATURE_HEIGHT, N_FEATURE_WIDTH) だが、実際は1次元配列。
	void  make_input_features(float* dst) const {
		std::fill_n(dst, N_FEATURE_WHC, 0);    // ひとまず全部0 埋めする。後から必要な箇所だけフラグを立てる。

		auto occupied_bb = pos.occupiedBB();

		//// 盤上の駒
		FOREACH_BB(occupied_bb, Square sq, {
			const Piece pc = pos.piece(sq);
			const PieceType pt = pieceToPieceType(pc);
			Color c = pieceToColor(pc);

			// 手番側の情報が来るチャンネルの位置の固定(前半のチャンネルに来るように)、
			// 手番側の駒を見る視点の固定(手番側を、便宜上の先手として盤面を見る)
			// (つまり、手番側が常に便宜上の先手となるように)
			const Color c2 = (turn() == Black ? c : oppositeColor(c));
			const Square sq2 = (turn() == Black ? sq : SQ99 - sq);

			set_feature(dst, c2, pt, sq2);
			});

		//// 持ち駒
		for (Color c = Black; c < ColorNum; ++c) {
			const Hand hand = pos.hand(c);
			for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp) {
				u32 num = hand.numOf(hp);
				num = std::min(num, N_MAX_HANDS[hp]);    // 上限値でclip

				// 手番側の情報が来るチャンネルの位置の固定(前半のチャンネルに来るように)
				// (つまり、手番側が常に便宜上の先手となるように)
				const Color c2 = (turn() == Black ? c : oppositeColor(c));

				set_feature_hand(dst, c2, hp, num);
			}
		}
	}

#elif defined(PYTHON_DLSHOGI2)
	// 盤上の駒について、フラグを立てる。
	// 色がcのplyaer の駒pt がsq に存在するというフラグを立てる。
	static void set_feature(float* dst, const Color c, const PieceType pt, const Square sq) {
		// PieceType は、occupied == 0, Pawn == 1, Lnace == 2, ...となっている。
		// 今回はoccupied は考慮しないので、その分を引いて(pt -1)としている。
		const u32 idx = (u32)c * N_FEATURE1_CHANNEL_PER_COLOR * N_FEATURE_WH + (u32)(pt - 1) * N_FEATURE_WH + (u32)sq;
		dst[idx] = 1;
	}


	// 持ち駒について、フラグを立てる。
	// num枚のチャンネル全てにフラグを立てる。(81 * num個のフラグを立てる。)
	// @arg c: 与えられた持ち駒hp を持っている方のplayerの色
	// @arg hp: 持ち駒の種類
	// @arg num: 持ち駒の数
	static void set_feature_hand(float* dst, const Color c, const HandPiece hp, const u32 num) {
		// 引数で渡されたhp の手前までの、各駒の持ち駒の上限数の和。
		// つまり、hp の持ち駒についてのチャンネルの先頭のchannel index.(チャンネル番号)
		const u32 tmp = std::accumulate(std::begin(N_MAX_HANDS), std::next(std::begin(N_MAX_HANDS), (u32)hp), 0);
		const u32 idx = (u32)ColorNum * N_FEATURE1_CHANNEL_PER_COLOR * N_FEATURE_WH
			+ (u32)c * N_FEATURE2_CHANNEL_PER_COLOR * N_FEATURE_WH + (tmp)*N_FEATURE_WH;
		std::fill_n(dst + idx, N_FEATURE_WH * num, 1);
	}

	// NOTE
	//     : 因みに、出来上がった配列の、例えば初期局面の先手の歩についてを見てみると、
	//       0 0 0 0 0 0 1 0 0
	//       0 0 0 0 0 0 1 0 0
	//       0 0 0 0 0 0 1 0 0
	//       0 0 0 0 0 0 1 0 0
	//       0 0 0 0 0 0 1 0 0
	//       0 0 0 0 0 0 1 0 0
	//       0 0 0 0 0 0 1 0 0
	//       0 0 0 0 0 0 1 0 0
	//       0 0 0 0 0 0 1 0 0
	//      となっている。
	// @arg dst
	//     : 形式上は(N_FEATURE_CHANNEL, N_FEATURE_HEIGHT, N_FEATURE_WIDTH) だが、実際は1次元配列。
	void  make_input_features(float* dst) const {
		std::fill_n(dst, N_FEATURE_WHC, 0);    // ひとまず全部0 埋めする。後から必要な箇所だけフラグを立てる。

		auto occupied_bb = pos.occupiedBB();

		//// 盤上の駒
		FOREACH_BB(occupied_bb, Square sq, {
			const Piece pc = pos.piece(sq);
			const PieceType pt = pieceToPieceType(pc);
			Color c = pieceToColor(pc);

			// 手番側の情報が来るチャンネルの位置の固定(前半のチャンネルに来るように)、
			// 手番側の駒を見る視点の固定(手番側を、便宜上の先手として盤面を見る)
			// (つまり、手番側が常に便宜上の先手となるように)
			const Color c2 = (turn() == Black ? c : oppositeColor(c));
			const Square sq2 = (turn() == Black ? sq : SQ99 - sq);

			set_feature(dst, c2, pt, sq2);
			});

		//// 持ち駒
		for (Color c = Black; c < ColorNum; ++c) {
			const Hand hand = pos.hand(c);
			for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp) {
				u32 num = hand.numOf(hp);
				num = std::min(num, N_MAX_HANDS[hp]);    // 上限値でclip

				// 手番側の情報が来るチャンネルの位置の固定(前半のチャンネルに来るように)
				// (つまり、手番側が常に便宜上の先手となるように)
				const Color c2 = (turn() == Black ? c : oppositeColor(c));

				set_feature_hand(dst, c2, hp, num);
			}
		}
	}
#elif defined(FEATURE_V1)

	// 盤上の駒について、フラグを立てる。
	// 色がcのplyaer の駒pt がsq に存在するというフラグを立てる。
	static void set_feature1(float* dst, const Color c, const PieceType pt, const Square sq) {
		// NOTE
		//     : PieceType は、occupied == 0, Pawn == 1, Lnace == 2, ...となっている。
		//       今回はoccupied は使わないので、その分を引いて(pt -1)としている。
		//const u32 idx = (u32)c * N_FEATURE_WHC_PER_COLOR + (u32)(pt - 1) * N_FEATURE_WH + (u32)sq;
		//const u32 idx =
		//	0                                      // feature1 より前の特徴量の総数
		//	+ (u32)c * N_FEATURE1_WHC_PER_COLOR
		//	+ (u32)(pt - 1) * N_FEATURE_WH
		//	+ (u32)sq;
		const u32&& idx =
			(
				0 +
				(u32)c * N_FEATURE1_CHANNEL_PER_COLOR +
				(u32)(pt - 1)
				) * N_FEATURE_WH + sq;
		dst[idx] = 1;
	}

	// 持ち駒について、フラグを立てる。
	// num枚のチャンネル全てにフラグを立てる。(81 * num個のフラグを立てる。)
	// @arg c: 与えられた持ち駒hp を持っている方のplayerの色
	// @arg hp: 持ち駒の種類
	// @arg num: 持ち駒の数
	static void set_feature2(float* dst, const Color c, const HandPiece hp, const u32 num) {
		// 引数で渡されたhp の手前までの、各駒の持ち駒の上限数の和。
		// つまり、hp の持ち駒についてのチャンネルの先頭のchannel index.(チャンネル番号)
		const u32&& hp_channel = std::accumulate(std::begin(N_MAX_HANDS), std::next(std::begin(N_MAX_HANDS), (u32)hp), 0);
		//const u32 idx = (u32)c * N_FEATURE_WHC_PER_COLOR + (N_FEATURE1_CHANNEL_PER_COLOR + tmp) * N_FEATURE_WH;
		//const u32 idx =
		//	N_FEATURE1_WHC         // feature1 より前の特徴量の総数
		//	+ (u32)c * N_FEATURE2_WHC_PER_COLOR
		//	+ hp_channel_start_idx * N_FEATURE_WH;
		const u32&& idx = (
			N_FEATURE1_CHANNEL +
			(u32)c * N_FEATURE2_CHANNEL_PER_COLOR +
			hp_channel
			) * N_FEATURE_WH;

		std::fill_n(dst + idx, N_FEATURE_WH * num, 1);
	}

	// 入力特徴量全体が、先に0初期化されているならば、in_check = true の時だけstore すれば良い。
	template<bool in_check>
	static void set_feature3(float* dst) {
		if constexpr (in_check) {
			constexpr u32 idx = (
				N_FEATURE1_CHANNEL + N_FEATURE2_CHANNEL
				) * N_FEATURE_WH;
			std::fill_n(dst + idx, N_FEATURE_WH, 1);
		}
		else {
			throw std::runtime_error();
		}
	}

	// @arg dst
	//     : 形式上は(N_FEATURE_CHANNEL, N_FEATURE_HEIGHT, N_FEATURE_WIDTH) だが、実際は1次元配列。
	void  make_input_features(float* dst) const {
		std::fill_n(dst, N_FEATURE_WHC, 0);    // ひとまず全部0 埋めする。後から必要な箇所だけフラグを立てる。

		auto occupied_bb = pos.occupiedBB();

		//// 盤上の駒
		FOREACH_BB(occupied_bb, Square sq, {
			const Piece pc = pos.piece(sq);
			const PieceType pt = pieceToPieceType(pc);
			Color c = pieceToColor(pc);

			// 手番側の情報が来るチャンネルの位置の固定(前半のチャンネルに来るように)、
			// 手番側の駒を見る視点の固定(手番側を、便宜上の先手として盤面を見る)
			// (つまり、手番側が常に便宜上の先手となるように)
			const Color c2 = (turn() == Black ? c : oppositeColor(c));
			const Square sq2 = (turn() == Black ? sq : SQ99 - sq);

			set_feature1(dst, c2, pt, sq2);
			});

		//// 持ち駒
		for (Color c = Black; c < ColorNum; ++c) {
			const Hand hand = pos.hand(c);
			for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp) {
				u32 num = hand.numOf(hp);
				num = std::min(num, N_MAX_HANDS[hp]);    // 上限値でclip

				// 手番側の情報が来るチャンネルの位置の固定(前半のチャンネルに来るように)
				// (つまり、手番側が常に便宜上の先手となるように)
				const Color c2 = (turn() == Black ? c : oppositeColor(c));

				set_feature2(dst, c2, hp, num);
			}
		}

		//// 王手
		if (pos.inCheck()) {
			set_feature3<true>(dst);
		}
	}

#elif defined(FEATURE_V2)

    // NOTE
    //     : 合計119枚の、9 * 9 画像を入力特徴量としている。
    //       https://tadaoyamaoka.hatenablog.com/entry/2022/01/10/212539#:~:text=%E7%8F%BE%E7%8A%B6%E3%81%AEdlshogi%E3%81%A7%E3%81%AF%E3%80%81%E5%85%A5%E5%8A%9B,%E9%87%8F%E3%81%AF1bit%E3%81%A7%E8%A1%A8%E3%81%9B%E3%82%8B%E3%80%82
    //     : 31(MAX_FEATURES1_NUM) * 2(ColorNum) = 62枚
    //     : 57枚
    typedef float features1_t[ColorNum][MAX_FEATURES1_NUM][SquareNum];
    typedef float features2_t[MAX_FEATURES2_NUM][SquareNum];
	typedef char packed_features1_t[((size_t)ColorNum * MAX_FEATURES1_NUM * (size_t)SquareNum + 7) / 8];
	typedef char packed_features2_t[((size_t)MAX_FEATURES2_NUM + 7) / 8];

	// HACK: 私の古い実装との互換性を保ったままにしようとしたせいで、割と無理のある感じになってる。
	//       全ての所で、dlshogi と同様に配列なtypedef を使うべき。

    //// dlshogi cppshogi.cpp より
	template<typename T, std::enable_if_t<std::is_same_v<T, features1_t>, std::nullptr_t> = nullptr>
    inline void set_features1(T* const features1, const Color c, const int f1idx, const Square sq) const {
    	(*features1)[c][f1idx][sq] = 1.0f;
    }
	template<typename T, std::enable_if_t<std::is_same_v<T, packed_features1_t>, std::nullptr_t> = nullptr>
	inline void set_features1(T* const packed_features1, const Color c, const int f1idx, const Square sq) const {
    	const int idx = MAX_FEATURES1_NUM * (int)SquareNum * (int)c + (int)SquareNum * f1idx + sq;
    	(*packed_features1)[idx >> 3] |= (1 << (idx & 7));
    }
    
	template<typename T, std::enable_if_t<std::is_same_v<T, features2_t>, std::nullptr_t> = nullptr>
	inline void set_features2(T* const features2, const Color c, const int f2idx, const u32 num) const {
    	std::fill_n((*features2)[MAX_PIECES_IN_HAND_SUM * (int)c + f2idx], (int)SquareNum * num, 1.0f);
    }
	template<typename T, std::enable_if_t<std::is_same_v<T, packed_features2_t>, std::nullptr_t> = nullptr>
	inline void set_features2(T* const packed_features2, const Color c, const int f2idx, const u32 num) const {
    	for (u32 i = 0; i < num; ++i) {
    		const int idx = MAX_PIECES_IN_HAND_SUM * (int)c + f2idx + i;
    		(*packed_features2)[idx >> 3] |= (1 << (idx & 7));
    	}
    }
    
	// 王手か否か
	template<typename T, std::enable_if_t<std::is_same_v<T, features2_t>, std::nullptr_t> = nullptr>
	inline void set_features2(T* const features2, const int f2idx) const {
    	std::fill_n((*features2)[f2idx], SquareNum, 1.0f);
    }
	template<typename T, std::enable_if_t<std::is_same_v<T, packed_features2_t>, std::nullptr_t> = nullptr>
	inline void set_features2(T* const packed_features2, const int f2idx) const {
    	(*packed_features2)[f2idx >> 3] |= (1 << (f2idx & 7));
    }

    // NOTE
    //     : dlshogi のmake_input_features に相当。
    // 駒の利き、王手情報を含む特徴量(dlshogi互換)
    template<typename T, bool use_set_feature = false>
    void make_input_features(T* mem1, T* mem2) const {
		if constexpr (std::is_same_v<T, float>) {
			// https://stackoverflow.com/questions/1143262/what-is-the-difference-between-const-int-const-int-const-and-int-const
			features1_t* const features1 = reinterpret_cast<features1_t* const>(mem1);
			features2_t* const features2 = reinterpret_cast<features2_t* const>(mem2);
			float(* const features2_hand)[ColorNum][MAX_PIECES_IN_HAND_SUM][SquareNum] = reinterpret_cast<float(* const)[ColorNum][MAX_PIECES_IN_HAND_SUM][SquareNum]>(mem2);

			// NOTE
			//     : 0 初期化
			std::fill_n((T*)features1, sizeof(features1_t) / sizeof(T), 0);
			std::fill_n((T*)features2, sizeof(features2_t) / sizeof(T), 0);

			const Bitboard occupied_bb = pos.occupiedBB();

			// NOTE:
			//     各駒による利きのbitboard の2次元配列 (0 で全てのbitboard を初期化している)
			// 駒の利き(駒種でマージ)
			Bitboard attacks[ColorNum][PieceTypeNum] = {
				{ { 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 } },
				{ { 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 } },
			};
			for (Square sq = SQ11; sq < SquareNum; sq++) {
				const Piece p = pos.piece(sq);
				if (p != Empty) {
					const Color pc = pieceToColor(p);
					const PieceType pt = pieceToPieceType(p);
					// NOTE
					//     : 位置sq にある手番pc の駒pt について、何かしらの駒がある箇所がoccupied_bb の時の利きのbb
					const Bitboard bb = pos.attacksFrom(pt, pc, sq, occupied_bb);
					attacks[pc][pt] |= bb;
				}
			}

			for (Color c = Black; c < ColorNum; ++c) {
				// NOTE
				//     : pos.turn() とは、局面自体の手番。
				//       c は、あくまでこれから見ようとする駒の手番。
				//       -> つまり、pos.turn() == White の時、c2は、cが黒の時白、白の時黒になる。
				//       -> もっと言うと、入力特徴量は、index == 0に手番側の駒の位置を、index == 1には敵の駒の位置を格納する。
				//          なので、
				//          pos.turn() == Black の時、黒(先手)の駒の位置がindex == 0に、白(後手)の駒の位置がindex == 1に、
				//          pos.turn() == White の時、白(後手)の駒の位置がindex == 0に、黒(先手)の駒の位置がindex == 1に、格納される。
				// 白の場合、色を反転
				const Color c2 = pos.turn() == Black ? c : oppositeColor(c);

				// NOTE
				//     : posより、各コマの配置をbitboard に格納。
				//       先後の区別はしない。
				//       つまり、PieceTypeNum個のbitboardの配列となる。
				// 駒の配置
				Bitboard bb[PieceTypeNum];
				for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
					// NOTE: pos から、color==c の各駒のbitboard を取得
					bb[pt] = pos.bbOf(pt, c);
				}

				for (Square sq = SQ11; sq < SquareNum; ++sq) {
					// NOTE
					//     : アクセスするインデックスを変更することで、実質的に回転している。
					// 白の場合、盤面を180度回転
					const Square sq2 = pos.turn() == Black ? sq : SQ99 - sq;

					for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
						// 駒の配置
						if (bb[pt].isSet(sq)) {

							if constexpr (use_set_feature) {
								set_features1(features1, c2, pt - 1, sq2);
							}
							else {
								(*features1)[c2][pt - 1][sq2] = 1;
							}
						}

						// 駒の利き
						if (attacks[c][pt].isSet(sq)) {
							if constexpr (use_set_feature) {
								set_features1(features1, c2, PIECETYPE_NUM + pt - 1, sq2);
							}
							else {
								(*features1)[c2][PIECETYPE_NUM + pt - 1][sq2] = 1;
							}
						}
					}

					// NOTE
					//     : c, sq で計算し、c2, sq2 でset_feature しているが問題ない。
					//       (c, sq2 やc2, sq の組み合わせで使わなければ問題ない)
					// 利き数
					const int num = std::min(MAX_ATTACK_NUM, pos.attackersTo(c, sq, occupied_bb).popCount());
					for (int k = 0; k < num; k++) {
						if constexpr (use_set_feature) {
							set_features1(features1, c2, PIECETYPE_NUM + PIECETYPE_NUM + k, sq2);
						}
						else {
							(*features1)[c2][PIECETYPE_NUM + PIECETYPE_NUM + k][sq2] = 1;
						}
					}
				}
				// hand
				const Hand hand = pos.hand(c);
				int p = 0;    // 現在処理している持ち駒に対するチャンネルの先頭のindex
				for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp) {
					u32 num = hand.numOf(hp);
					if (num >= MAX_PIECES_IN_HAND[hp]) {
						// 上限値でclip
						num = MAX_PIECES_IN_HAND[hp];
					}

					if constexpr (use_set_feature) {
						set_features2(features2, c2, p, num);
					}
					else {
						std::fill_n((*features2_hand)[c2][p], (int)SquareNum * num, 1);
					}
					p += MAX_PIECES_IN_HAND[hp];
				}
			}

			// is check
			if (pos.inCheck()) {
				if constexpr (use_set_feature) {
					set_features2(features2, MAX_FEATURES2_HAND_NUM);
				}
				else {
					std::fill_n((*features2)[MAX_FEATURES2_HAND_NUM], SquareNum, 1);
				}
			}
		}
		else if constexpr (std::is_same_v<T, char>) {
			//static_assert(true_v<T>, "true");
			//static_assert(true, "true");


			packed_features1_t* const features1 = reinterpret_cast<packed_features1_t* const>(mem1);
			packed_features2_t* const features2 = reinterpret_cast<packed_features2_t* const>(mem2);
			//float(* const features2_hand)[ColorNum][MAX_PIECES_IN_HAND_SUM][SquareNum] =
			//    reinterpret_cast<float(* const)[ColorNum][MAX_PIECES_IN_HAND_SUM][SquareNum]>(mem2);

			// NOTE
			//     : 0 初期化
			std::fill_n((T*)features1, sizeof(packed_features1_t) / sizeof(T), 0);
			std::fill_n((T*)features2, sizeof(packed_features2_t) / sizeof(T), 0);

			const Bitboard occupied_bb = pos.occupiedBB();

			// NOTE:
			//     各駒による利きのbitboard の2次元配列 (0 で全てのbitboard を初期化している)
			// 駒の利き(駒種でマージ)
			Bitboard attacks[ColorNum][PieceTypeNum] = {
				{ { 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 } },
				{ { 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 } },
			};
			for (Square sq = SQ11; sq < SquareNum; sq++) {
				const Piece p = pos.piece(sq);
				if (p != Empty) {
					const Color pc = pieceToColor(p);
					const PieceType pt = pieceToPieceType(p);
					// NOTE
					//     : 位置sq にある手番pc の駒pt について、何かしらの駒がある箇所がoccupied_bb の時の利きのbb
					const Bitboard bb = pos.attacksFrom(pt, pc, sq, occupied_bb);
					attacks[pc][pt] |= bb;
				}
			}

			for (Color c = Black; c < ColorNum; ++c) {
				// NOTE
				//     : pos.turn() とは、局面自体の手番。
				//       c は、あくまでこれから見ようとする駒の手番。
				//       -> つまり、pos.turn() == White の時、c2は、cが黒の時白、白の時黒になる。
				//       -> もっと言うと、入力特徴量は、index == 0に手番側の駒の位置を、index == 1には敵の駒の位置を格納する。
				//          なので、
				//          pos.turn() == Black の時、黒(先手)の駒の位置がindex == 0に、白(後手)の駒の位置がindex == 1に、
				//          pos.turn() == White の時、白(後手)の駒の位置がindex == 0に、黒(先手)の駒の位置がindex == 1に、格納される。
				// 白の場合、色を反転
				const Color c2 = pos.turn() == Black ? c : oppositeColor(c);

				// NOTE
				//     : posより、各コマの配置をbitboard に格納。
				//       先後の区別はしない。
				//       つまり、PieceTypeNum個のbitboardの配列となる。
				// 駒の配置
				Bitboard bb[PieceTypeNum];
				for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
					// NOTE: pos から、color==c の各駒のbitboard を取得
					bb[pt] = pos.bbOf(pt, c);
				}

				for (Square sq = SQ11; sq < SquareNum; ++sq) {
					// NOTE
					//     : アクセスするインデックスを変更することで、実質的に回転している。
					// 白の場合、盤面を180度回転
					const Square sq2 = pos.turn() == Black ? sq : SQ99 - sq;

					for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
						// 駒の配置
						if (bb[pt].isSet(sq)) {
							set_features1(features1, c2, pt - 1, sq2);
						}

						// 駒の利き
						if (attacks[c][pt].isSet(sq)) {
							set_features1(features1, c2, PIECETYPE_NUM + pt - 1, sq2);
						}
					}

					// NOTE
					//     : c, sq で計算し、c2, sq2 でset_feature しているが問題ない。
					//       (c, sq2 やc2, sq の組み合わせで使わなければ問題ない)
					// 利き数
					const int num = std::min(MAX_ATTACK_NUM, pos.attackersTo(c, sq, occupied_bb).popCount());
					for (int k = 0; k < num; k++) {
						set_features1(features1, c2, PIECETYPE_NUM + PIECETYPE_NUM + k, sq2);
					}
				}
				// hand
				const Hand hand = pos.hand(c);
				int p = 0;    // 現在処理している持ち駒に対するチャンネルの先頭のindex
				for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp) {
					u32 num = hand.numOf(hp);
					if (num >= MAX_PIECES_IN_HAND[hp]) {
						// 上限値でclip
						num = MAX_PIECES_IN_HAND[hp];
					}
					set_features2(features2, c2, p, num);
					p += MAX_PIECES_IN_HAND[hp];
				}
			}

			// is check
			if (pos.inCheck()) {
				set_features2(features2, MAX_FEATURES2_HAND_NUM);
			}
		}
    }
#else    // どの特徴量も定義されていない時
	// 
	void  make_input_features(float* _dst) const {
		std::cout << "Error: feature_v is not defined." << std::endl;
	}
#endif    // NOT_IMPLEMENTED
	

#if defined (FEATURE_V2)
	// python から呼び出す用。
	// c++ から呼び出してはならない。
	void  __make_input_features(char* _dst1, char* _dst2) const {
		float* dst1 = reinterpret_cast<float*>(_dst1);
		float* dst2 = reinterpret_cast<float*>(_dst2);
		make_input_features(dst1, dst2);
	}

#if not defined(FOR_BUILD_CSHOGI)
	// HACK: ぱっと見ポインタにも参照にも見えへんの、微妙やなぁ～
	// 探索時に呼び出す
	void make_input_features(feature_t feature) const {
		make_input_features(std::get<0>(feature), std::get<1>(feature));
	}
#endif

#else
	// python から呼び出す
	void  make_input_features(char* _dst) const {
		float* dst = reinterpret_cast<float*>(_dst);
		make_input_features(dst);
	}


#if not defined(FOR_BUILD_CSHOGI)
	//// HACK: ぱっと見ポインタにも参照にも見えへんの、微妙やなぁ～
	//// typedef はあれかもな、マクロ的な扱いかもな。知らんけど。
	//// 探索時に呼び出す
	//void make_input_features(feature_t _feature) const {
	//	float* feature = reinterpret_cast<float*>(_feature);
	//	make_input_features(feature);
	//}
#endif

#endif

	unsigned long long bookKey() {
		return Book::bookKey(pos);
	}

	Position pos;


private:
	// 一時的にpublic に。
	std::deque<std::pair<Move, StateInfo>> history;

	void bbToVector(PieceType pt, Color c, Piece piece, std::vector<int>& board) const {
		Bitboard bb = pos.bbOf(pt, c);
		while (bb) {
			const Square sq = bb.firstOneFromSQ11();
			board[sq] = piece;
		}
	}
};

// Move to label_idx
inline int move_to_label() {

}

class __LegalMoveList
{
public:
	__LegalMoveList() {}
	__LegalMoveList(const __Board& board) {
		// TODO: Legal に。
		ml.reset(new MoveList<Legal>(board.pos));
	}

	bool end() const { return ml->end(); }
	int move() const { return ml->move().value(); }
	Move Move() const { return ml->move(); }
	void next() { ++(*ml); }
	int size() const { return (int)ml->size(); }

private:
	// TODO: Legal に。
	std::shared_ptr<MoveList<Legal>> ml;
};

class __PseudoLegalMoveList
{
public:
	__PseudoLegalMoveList() {}
	__PseudoLegalMoveList(const __Board& board) {
		ml.reset(new MoveList<PseudoLegal>(board.pos));
	}

	bool end() const { return ml->end(); }
	int move() const { return ml->move().value(); }
	void next() { ++(*ml); }
	int size() const { return (int)ml->size(); }

private:
	std::shared_ptr<MoveList<PseudoLegal>> ml;
};

inline int __piece_to_piece_type(const int p) { return (int)pieceToPieceType((Piece)p); }
inline int __hand_piece_to_piece_type(const int hp) { return (int)handPieceToPieceType((HandPiece)hp); }

// 移動先
inline int __move_to(const int move) { return (move >> 0) & 0x7f; }
// 移動元
inline int __move_from(const int move) { return (move >> 7) & 0x7f; }
// 取った駒の種類
inline int __move_cap(const int move) { return (move >> 20) & 0xf; }
// 成るかどうか
inline bool __move_is_promotion(const int move) { return move & Move::PromoteFlag; }
// 駒打ちか
inline bool __move_is_drop(const int move) { return __move_from(move) >= 81; }
// 移動する駒の種類
inline int __move_from_piece_type(const int move) { return (move >> 16) & 0xf; };
// 打つ駒の種類
inline int __move_drop_hand_piece(const int move) { return pieceTypeToHandPiece((PieceType)__move_from(move) - SquareNum + 1); }

inline unsigned short __move16(const int move) { return (unsigned short)move; }

inline unsigned short __move16_from_psv(const unsigned short move16) {
	const unsigned short MOVE_DROP = 1 << 14;
	const unsigned short MOVE_PROMOTE = 1 << 15;

	unsigned short to = move16 & 0x7f;
	unsigned short from = (move16 >> 7) & 0x7f;
	if ((move16 & MOVE_DROP) != 0) {
		from += SquareNum - 1;
	}
	return to | (from << 7) | ((move16 & MOVE_PROMOTE) != 0 ? Move::PromoteFlag : 0);
}

inline unsigned short __move16_to_psv(const unsigned short move16) {
	const unsigned short MOVE_DROP = 1 << 14;
	const unsigned short MOVE_PROMOTE = 1 << 15;

	unsigned short to = move16 & 0x7f;
	unsigned short from = (move16 >> 7) & 0x7f;
	unsigned short drop = 0;
	if (from >= 81) {
		from -= SquareNum - 1;
		drop = MOVE_DROP;
	}
	return to | (from << 7) | drop | ((move16 & Move::PromoteFlag) != 0 ? MOVE_PROMOTE : 0);
}

// 反転
inline int __move_rotate(const int move) {
	int to = __move_to(move);
	to = SQ99 - to;
	int from = __move_from(move);
	if (!__move_is_drop(move))
		from = SQ99 - from;
	return (move & 0xffff0000) | to | (from << 7);
}

inline std::string __move_to_usi(const int move) { return Move(move).toUSI(); }
inline std::string __move_to_csa(const int move) { return Move(move).toCSA(); }

// NOTE: const in& dir_x, にしようぜ
inline MOVE_DIRECTION get_move_direction(const int dir_x, const int dir_y) {
	if (dir_y < 0 && dir_x == 0) {
		return UP;
	}
	else if (dir_y == -2 && dir_x == -1) {
		return UP2_LEFT;
	}
	else if (dir_y == -2 && dir_x == 1) {
		return UP2_RIGHT;
	}
	else if (dir_y < 0 && dir_x < 0) {
		return UP_LEFT;
	}
	else if (dir_y < 0 && dir_x > 0) {
		return UP_RIGHT;
	}
	else if (dir_y == 0 && dir_x < 0) {
		return LEFT;
	}
	else if (dir_y == 0 && dir_x > 0) {
		return RIGHT;
	}
	else if (dir_y > 0 && dir_x == 0) {
		return DOWN;
	}
	else if (dir_y > 0 && dir_x < 0) {
		return DOWN_LEFT;
	}
	else /* if (dir_y > 0 && dir_x > 0) */ {
		return DOWN_RIGHT;
	}
}

// 駒の移動を表すラベル(dlshogi互換)
// @arg move: Move m; について、m.proFromAndTo() の戻り値に相当するもの。
inline int __dlshogi_make_move_label(const int move, const int color) {
	// see: move.hpp : 30
	// xxxxxxxx x1111111  移動先
	// xx111111 1xxxxxxx  移動元。駒打ちの際には、PieceType + SquareNum - 1
	// x1xxxxxx xxxxxxxx  1 なら成り
	u16 to_sq = move & 0b1111111;
	u16 from_sq = (move >> 7) & 0b1111111;

	if (from_sq < SquareNum) {    // NOTE: 
		// 白の場合、盤面を180度回転
		if (color == White) {
			to_sq = (u16)SQ99 - to_sq;
			from_sq = (u16)SQ99 - from_sq;
		}

		// NOTE
		//     : http://www.bohyoh.com/CandCPP/C/Library/div_t.html
		//     : x,y は、SQab に対して、x = a - 1, y = b - 1 となる。
		//       例えば、SQ19なら、x = 0, y = 8。
		//       (後手番の左下を原点とした時の座標)
		const div_t to_d = div(to_sq, 9);
		const int to_x = to_d.quot;    // NOTE: 商
		const int to_y = to_d.rem;     // NOTE: 剰余
		const div_t from_d = div(from_sq, 9);
		const int from_x = from_d.quot;
		const int from_y = from_d.rem;
		const int dir_x = from_x - to_x;
		const int dir_y = to_y - from_y;

		// 移動の前後での座標の変化から、移動の向きを計算。
		MOVE_DIRECTION move_direction = get_move_direction(dir_x, dir_y);

		// promote
		if ((move & 0b100000000000000) > 0) {
			// NOTE
			//     : 10 は移動方向の総数(8方向 + 桂馬の2方向)
			//       ※各移動方向には、移動して且つ不成 & 移動して成る の2種類がある。
			move_direction = (MOVE_DIRECTION)(move_direction + 10);
		}
		return 9 * 9 * move_direction + to_sq;
	}
	// 持ち駒の場合
	else {
		// 白の場合、盤面を180度回転
		if (color == White) {
			to_sq = (u16)SQ99 - to_sq;
		}
		// NOTE
		//     : ※hand_piece は計7種類
		//     : つまり、移動のラベルは、81 * (10 + 10 + 7)
		const int hand_piece = from_sq - (int)SquareNum;
		const int move_direction_label = MOVE_DIRECTION_NUM + hand_piece;
		return 9 * 9 * move_direction_label + to_sq;
	}
}

// HACK
//     : 変なこだわりで、引数の型をMoveにしたいが為に、
//       違う名前なのにやってることが全く一緒な関数が出来てて、ちょっと気持ち悪くなってる。
//       せめて、__dlshogi_make_move_label() もmove_to_label() にして多重定義するなりしようぜ。
// 現状、中身は__dlshogi_make_move_label() と同じ。
// @arg move
//     : Move のメソッドが使えるのでMove で受け取っているが、
//       下位15bit あれば事足りる。dlshogiでは、move16 とも呼ばれている。
//       (下位15bit あれば事足りるのは、駒うちの時しか駒タイプを必要としていないから。)
inline int move_to_label(const Move& move, const Color& color) {
	u16&& to_sq = move.to();
	u16&& from_sq = move.from();

	if (from_sq < SquareNum) {
		// 白の場合、盤面を180度回転
		if (color == White) {
			to_sq = (u16)SQ99 - to_sq;
			from_sq = (u16)SQ99 - from_sq;
		}

		// NOTE
		//     : http://www.bohyoh.com/CandCPP/C/Library/div_t.html
		//     : x,y は、SQab に対して、x = a - 1, y = b - 1 となる。
		//       例えば、SQ19なら、x = 0, y = 8。
		//       (後手番の左下を原点とした時の座標)
#if false
		const div_t to_d = div(to_sq, 9);
		const int to_x = to_d.quot;
		const int to_y = to_d.rem;
		const div_t from_d = div(from_sq, 9);
		const int from_x = from_d.quot;
		const int from_y = from_d.rem;
#else
		const div_t& to_d = div(to_sq, 9);
		const div_t& from_d = div(from_sq, 9);
		const int& to_x = to_d.quot;       // NOTE: 商
		const int& from_x = from_d.quot;   // NOTE: 商
		const int& to_y = to_d.rem;        // NOTE: 剰余
		const int& from_y = from_d.rem;    // NOTE: 剰余
#endif
		const int&& dir_x = from_x - to_x;
		const int&& dir_y = to_y - from_y;

		MOVE_DIRECTION&& move_direction = get_move_direction(dir_x, dir_y);

		// promote
		if (move.isPromotion()) {
			move_direction = (MOVE_DIRECTION)(move_direction + N_LABEL_WITHOUT_PRO);
		}
		return move_direction * N_LABEL_SQ_PER_CHANNEL + to_sq;
	}
	// 持ち駒の場合
	else {
		// 白の場合、盤面を180度回転
		if ((Color)color == White) {
			to_sq = (u16)SQ99 - to_sq;
		}

		// NOTE
		//     : move.pieceTypeDropped() - 1 としているのは、pieceType のpawn = 1 スタートなので、
		//       インデックス化するには -1 する。
		//     : 以下のmove.handPieceDropped(); と(int)move.pieceTypeDropped() - 1; は一見同じに見えるが違う。
		//       HGold, HBishop, HRook と Bishop, Rook, Gold となっていて、金、角、飛車の値が異なる。
		//       dlshogi に合わせるには、後者を採用する。python-dlshogi2 に合わせるには前者。
		//     : hand_piece はインデックス的
#if defined(FEATURE_V2)
		// dlshogi 互換
		const int&& hand_piece = (int)move.pieceTypeDropped() - 1;
#else
		// python-dlshogi2 互換。
		// 私のfeature_v0, v1 でもこちらを採用していた(はず。)
		const HandPiece&& hand_piece = move.handPieceDropped();
#endif

		const int move_direction_label = MOVE_DIRECTION_NUM + hand_piece;
		return move_direction_label * N_LABEL_SQ_PER_CHANNEL + to_sq;
	}
}

#if defined(DFPN_CSHOGI)
class __DfPn
{
public:
	__DfPn() {}
	__DfPn(const int max_depth, const uint32_t max_search_node, const int draw_ply)
		: dfpn(max_depth, max_search_node, draw_ply)
	{

	}
	bool search(__Board& board) {
		//std::cout << "debug: CSHOGI __DfPn::search()" << std::endl;
		pv.clear();
		return dfpn.dfpn(board.pos);
	}

	// NOTE: 自分が詰まされるか否かの探索かな？
	bool search_andnode(__Board& board) {
		return dfpn.dfpn_andnode(board.pos);
	}

	// NOTE: 探索を終了するか否かをset。つまり、set_is_stop() とも言える。
	void stop(bool is_stop) {
		dfpn.dfpn_stop(is_stop);
	}
	// NOTE: 詰ます手順の初手を返す
	int get_move(__Board& board) {
		return dfpn.dfpn_move(board.pos).value();
	}
	// NOTE: pv を、メンバのstd::vector<u32> pv; に格納する。
	void get_pv(__Board& board) {
		pv.clear();
		dfpn.get_pv(board.pos, pv);
	}

	//// NOTE: 探索時のoption を設定
	void set_draw_ply(const int draw_ply) {
		dfpn.set_draw_ply(draw_ply);
	}
	void set_maxdepth(const int depth) {
		dfpn.set_maxdepth(depth);
	}
	void set_max_search_node(const uint32_t max_search_node) {
		dfpn.set_max_search_node(max_search_node);
	}

	// NOTE: 合計探索数を返す？
	uint32_t get_searched_node() {
		return dfpn.searchedNode;
	}

	std::string get_pv_str(__Board& board) {
		this->get_pv(board);

		std::string pv_str = "";
		const auto&& pv_size = pv.size();
		if (pv_size > 0) {
			pv_str += __move_to_usi(pv[0]);
			for (int i = 1; i < pv_size; ++i) {
				auto&& move = pv[i];
				pv_str += " " + __move_to_usi(move);
			}
		}
		return pv_str;
	}

	std::vector<u32> pv;
private:
	DfPn dfpn;
};
#elif defined(DFPN_DLSHOGI)
class __DfPn
{
public:
	__DfPn() {}
	__DfPn(const int max_depth, const uint32_t max_search_node, const int draw_ply) {
		// NOTE: PvMateSearcher::PvMateSearcher() より
		dfpn.init();
		dfpn.set_maxdepth(max_depth);
		dfpn.set_max_search_node(max_search_node);
		dfpn.set_draw_ply(draw_ply);
	}
	bool search(__Board& board) {
		// NOTE; ひとまずここで毎度0初期化する
		dfpn.searchedNode = 0;
		//std::cout << "debug: DLSHOGI __DfPn::search()" << std::endl;
		//pv.clear();
		return dfpn.dfpn(board.pos);
	}

	// NOTE: 自分が詰まされるか否かの探索かな？
	bool search_andnode(__Board& board) {
		return dfpn.dfpn_andnode(board.pos);
	}

	// NOTE: 探索を終了するか否かをset。つまり、set_is_stop() とも言える。
	void stop(bool is_stop) {
		dfpn.dfpn_stop(is_stop);
	}
	// NOTE: 詰ます手順の初手を返す
	int get_move(__Board& board) {
		return dfpn.dfpn_move(board.pos).value();
	}
	// NOTE
	//     : これは恐らく、探索したnode の内、ある詰み手順が存在する任意のnode におけるpv が得られる。
	auto get_pv(__Board& board) {
		return dfpn.get_pv(board.pos);
	}

	//// NOTE: 探索時のoption を設定
	void set_hashsize(const uint64_t size) {
		dfpn.set_hashsize(size);
		// HACK: 本当はisready の時にのみ実行すべきだが、ひとまずはhash_size を変更するたびにresize する。
		dfpn.init();
	}
	void set_draw_ply(const int draw_ply) {
		dfpn.set_draw_ply(draw_ply);
	}
	void set_maxdepth(const int depth) {
		dfpn.set_maxdepth(depth);
	}
	void set_max_search_node(const uint32_t max_search_node) {
		dfpn.set_max_search_node(max_search_node);
	}

	// NOTE: 合計探索数を返す？
	uint32_t get_searched_node() {
		return dfpn.searchedNode;
	}

	std::string get_option_str() const {
		return dfpn.get_option_str();
	}

	std::string get_pv_str(__Board& board) {
		std::string pv_str;
		int depth;
		Move best_move;
		std::tie(pv_str, depth, best_move) = dfpn.get_pv(board.pos);
		return pv_str;
	}

	//std::vector<u32> pv;
private:
	DfPn dfpn;
};
#elif defined(DFPN_PARALLEL)
class __DfPn
{
public:
	__DfPn() {}
	__DfPn(const int max_depth, const uint32_t max_search_node, const int draw_ply) {
		// NOTE: PvMateSearcher::PvMateSearcher() より
		dfpn.init();
		dfpn.set_maxdepth(max_depth);
		dfpn.set_max_search_node(max_search_node);
		dfpn.set_draw_ply(draw_ply);
	}
	bool search(__Board& board) {
		// NOTE; ひとまずここで毎度0初期化する
		dfpn.searchedNode = 0;
		//std::cout << "debug: DLSHOGI __DfPn::search()" << std::endl;
		//pv.clear();
		return dfpn.dfpn(board.pos);
	}

	// NOTE: 自分が詰まされるか否かの探索かな？
	bool search_andnode(__Board& board) {
		return dfpn.dfpn_andnode(board.pos);
	}

	// NOTE: 探索を終了するか否かをset。つまり、set_is_stop() とも言える。
	void stop(bool is_stop) {
		dfpn.dfpn_stop(is_stop);
	}
	// NOTE: 詰ます手順の初手を返す
	int get_move(__Board& board) {
		return dfpn.dfpn_move(board.pos).value();
	}
	// NOTE
	//     : これは恐らく、探索したnode の内、ある詰み手順が存在する任意のnode におけるpv が得られる。
	auto get_pv(__Board& board) {
		return dfpn.get_pv(board.pos);
	}

	void reset() {
		dfpn.reset();
	}

	//// NOTE: 探索時のoption を設定
	void set_hashsize(const uint64_t size) {
		dfpn.set_hashsize(size);
		// HACK: 本当はisready の時にのみ実行すべきだが、ひとまずはhash_size を変更するたびにresize する。
		dfpn.init();
	}
	void set_draw_ply(const int draw_ply) {
		dfpn.set_draw_ply(draw_ply);
	}
	void set_maxdepth(const int depth) {
		dfpn.set_maxdepth(depth);
	}
	void set_max_search_node(const uint32_t max_search_node) {
		dfpn.set_max_search_node(max_search_node);
	}

	// NOTE: 合計探索数を返す？
	uint32_t get_searched_node() {
		return dfpn.searchedNode;
	}

	std::string get_option_str() const {
		return dfpn.get_option_str();
	}

	std::string get_pv_str(__Board& board) {
		std::string pv_str;
		int depth;
		Move best_move;
		std::tie(pv_str, depth, best_move) = dfpn.get_pv(board.pos);
		return pv_str;
	}

	//std::vector<u32> pv;
private:
	ParallelDfPn dfpn;
};
#elif defined(DFPN_PARALLEL2)
class __DfPn
{
public:
	__DfPn() {}
	__DfPn(const int max_depth, const uint32_t max_search_node, const int draw_ply) {
		// NOTE: PvMateSearcher::PvMateSearcher() より
		dfpn.init();
		dfpn.set_maxdepth(max_depth);
		dfpn.set_max_search_node(max_search_node);
		dfpn.set_draw_ply(draw_ply);
	}

	void new_search() {
		// 何もしない。何故なら、このdfpn は、dfpn::dfpn() 内でnew_search() してくれるからである。
	}

	bool search(__Board& board) {
		// NOTE; ひとまずここで毎度0初期化する
		dfpn.searchedNode = 0;
		//std::cout << "debug: DLSHOGI __DfPn::search()" << std::endl;
		//pv.clear();
		return dfpn.dfpn(board.pos);
	}

	// NOTE: 自分が詰まされるか否かの探索かな？
	bool search_andnode(__Board& board) {
		return dfpn.dfpn_andnode(board.pos);
	}

	// NOTE: 探索を終了するか否かをset。つまり、set_is_stop() とも言える。
	void stop(bool is_stop) {
		dfpn.dfpn_stop(is_stop);
	}
	// NOTE: 詰ます手順の初手を返す
	int get_move(__Board& board) {
		return dfpn.dfpn_move(board.pos).value();
	}
	// NOTE
	//     : これは恐らく、探索したnode の内、ある詰み手順が存在する任意のnode におけるpv が得られる。
	auto get_pv(__Board& board) {
		return dfpn.get_pv(board.pos);
	}

	void reset() {
		dfpn.reset();
	}

	//// NOTE: 探索時のoption を設定
	void set_hashsize(const uint64_t size) {
		dfpn.set_hashsize(size);
		// HACK: 本当はisready の時にのみ実行すべきだが、ひとまずはhash_size を変更するたびにresize する。
		dfpn.init();
	}
	void set_draw_ply(const int draw_ply) {
		dfpn.set_draw_ply(draw_ply);
	}
	void set_maxdepth(const int depth) {
		dfpn.set_maxdepth(depth);
	}
	void set_max_search_node(const uint32_t max_search_node) {
		dfpn.set_max_search_node(max_search_node);
	}

	// NOTE: 合計探索数を返す？
	uint32_t get_searched_node() {
		return dfpn.searchedNode;
	}

	std::string get_option_str() const {
		return dfpn.get_option_str();
	}

	std::string get_pv_str(__Board& board) {
		std::string pv_str;
		int depth;
		Move best_move;
		std::tie(pv_str, depth, best_move) = dfpn.get_pv(board.pos);
		return pv_str;
	}

	//std::vector<u32> pv;
private:
	ParallelDfPn dfpn;
};
#elif defined(DFPN_PARALLEL3)
class __DfPn
{
public:
	__DfPn() {}
	__DfPn(const int max_depth, const uint32_t max_search_node, const int draw_ply) {
		// NOTE: PvMateSearcher::PvMateSearcher() より
		dfpn.init();
		dfpn.set_maxdepth(max_depth);
		dfpn.set_max_search_node(max_search_node);
		dfpn.set_draw_ply(draw_ply);
	}

	void new_search() {
		dfpn.new_search();
	}

	bool search(__Board& board, int64_t& searched_node, const int threadid) {
		// NOTE; ひとまずここで毎度0初期化する
		//dfpn.searchedNode = 0;
		//std::cout << "debug: DLSHOGI __DfPn::search()" << std::endl;
		//pv.clear();
		return dfpn.dfpn(board.pos, searched_node, threadid);
	}

	// NOTE: 自分が詰まされるか否かの探索かな？
	bool search_andnode(__Board& board, int64_t& searched_node, const int threadid) {
		return dfpn.dfpn_andnode(board.pos, searched_node, threadid);
	}

	// NOTE: 探索を終了するか否かをset。つまり、set_is_stop() とも言える。
	void stop(bool is_stop) {
		dfpn.dfpn_stop(is_stop);
	}
	// NOTE: 詰ます手順の初手を返す
	int get_move(__Board& board) {
		return dfpn.dfpn_move(board.pos).value();
	}
	// NOTE
	//     : これは恐らく、探索したnode の内、ある詰み手順が存在する任意のnode におけるpv が得られる。
	auto get_pv(__Board& board) {
		return dfpn.get_pv(board.pos);
	}

	auto get_pv_safe(__Board& board) {
		return dfpn.get_pv<true>(board.pos);
	}

	void reset() {
		dfpn.reset();
	}

	//// NOTE: 探索時のoption を設定
	void set_hashsize(const uint64_t size) {
		dfpn.set_hashsize(size);
		// HACK: 本当はisready の時にのみ実行すべきだが、ひとまずはhash_size を変更するたびにresize する。
		dfpn.init();
	}
	void set_draw_ply(const int draw_ply) {
		dfpn.set_draw_ply(draw_ply);
	}
	void set_maxdepth(const int depth) {
		dfpn.set_maxdepth(depth);
	}
	void set_max_search_node(const uint32_t max_search_node) {
		dfpn.set_max_search_node(max_search_node);
	}

	//// NOTE: 合計探索数を返す？
	//uint32_t get_searched_node() {
	//	return dfpn.searchedNode;
	//}

	std::string get_option_str() const {
		return dfpn.get_option_str();
	}

	std::string get_pv_str(__Board& board) {
		std::string pv_str;
		int depth;
		Move best_move;
		std::tie(pv_str, depth, best_move) = dfpn.get_pv(board.pos);
		return pv_str;
	}

	template<bool or_node> void print_entry_info(__Board& board) {
		if constexpr (or_node) {
			dfpn.print_entry_info<true>(board.pos);
		}
		else {
			dfpn.print_entry_info<false>(board.pos);
		}
	}

	//std::vector<u32> pv;
private:
	ParallelDfPn dfpn;
};
#endif



#endif
