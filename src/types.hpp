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
// cppshogi/cppshogi.h ���
// ==================================================
struct HuffmanCodedPosAndEval2 {
	HuffmanCodedPos hcp;
	s16 eval;
	u16 bestMove16;
	uint8_t result; // xxxxxx11 : ���s�Axxxxx1xx : �����Axxxx1xxx : ���ʐ錾�Axxx1xxxx : �ő�萔
};
static_assert(sizeof(HuffmanCodedPosAndEval2) == 38, "");

struct HuffmanCodedPosAndEval3 {
	HuffmanCodedPos hcp; // �J�n�ǖ�
	u16 moveNum; // �萔
	u8 result; // xxxxxx11 : ���s�Axxxxx1xx : �����Axxxx1xxx : ���ʐ錾�Axxx1xxxx : �ő�萔
	u8 opponent; // �ΐ푊��i0:���ȑ΋ǁA1:���usi�A2:���usi�j
};
static_assert(sizeof(HuffmanCodedPosAndEval3) == 36, "");

struct MoveInfo {
	u16 selectedMove16; // �w����
	s16 eval; // �]���l
	u16 candidateNum; // ����̐�
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
	HuffmanCodedPos hcp; // �ǖ�
	int eval;
	float result;
	int count; // �d���J�E���g
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
	int eval;                                     // 200�Ƃ��A3000�Ƃ��A-1500 �݂����ȏ����]���l
	// �������猩�ėL���Ȃ�+, �s���Ȃ�-
	float result;                                 // ��ԑ�����݂ď����Ȃ�1, ���������Ȃ�0.5, �����Ȃ�0
	std::unordered_map<u16, float> candidates;    // NOTE
	//     : probability ��������ۂ��B
	//     ; hcpe �̂悤�ȍőP��݂̂��o�^���ꂽ���t�ł́A
	//       candidates[�őP��] = 1; �Ƃ����g���������Ă邪�A
	//       ����1 ���A���̎w����̊m�� = 100% �Ƒ����邱�Ƃ��o����B
	int count; // �d���J�E���g
};

constexpr u8 GAMERESULT_SENNICHITE = 0x4;
constexpr u8 GAMERESULT_NYUGYOKU = 0x8;
constexpr u8 GAMERESULT_MAXMOVE = 0x16;

// ==================================================
// �����Œ�`
// ==================================================

typedef struct Teacher {
	std::string sfen;
	int move;        // __move_to_usi() ���g���ƁAusi�`���ɒ������B
	int ply;        // sfen �Ɋ܂܂�Ă��邯�ǁA�ʂŊi�[�B
	float value;    // TODO: ����̎d�l���߂�
	float result;    // ��ԑ����猩�āA0�Ȃ畉���A1�Ȃ珟���A0.5�Ȃ��������

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

// HACK: YoBokkMoveInfo �Ƃ��̕����������肭�邩���B
// https://github.com/yaneurao/YaneuraOu/wiki/%E5%AE%9A%E8%B7%A1%E3%81%AE%E4%BD%9C%E6%88%90
// ��̌���ɂ��Ă̏��
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

// map ��key/value �Ō�����value �ɑ���������
// ����ꗗ��ێ��B
typedef struct YoBookValue {
	int ply;
	std::vector<yo_book_move_t> moves;     // �w���胊�X�g

	YoBookValue()
		: ply(-1)
	{

	}

	YoBookValue(const int& ply_)
		: ply(ply_)
	{

	}

	// HACK: get_best_eval_move() �Ƃ��̕����������肭��B
	// �]���l����ɍőP���Ԃ��B
	[[deprecated("Use get_best_eval_move() instead.")]]
	yo_book_move_t get_best_by_eval() const {
		return *std::max_element(moves.begin(), moves.end(), [](const auto& x, const auto& y) {
			return x.eval < y.eval;
		});
	}

	// �]���l����ԗǂ��A�X�ɂ��̒���depth ����ԍ����w����(:=�őP��)��eva_limit, depth_limit ���ɖ������ꍇ�݂̂����Ԃ��B
	// ��˂��牤�`���Ȃ�sort ����Ă�͂������ǁA�O�̈בS�T������(depth ���ɂ�sort ����ĂȂ������Ǝv���̂őS�T�����đ������͂��B�B�B)�B
	// @arg eval_limit
	//     : �őP�肪����ȏ�̕]���l �ł���Β�Ղɓo�^����Ă�őP���Ԃ��B
	//     : ���R�����A�����ڐ��̕]���l�ł���(�������L���Ȃ�+, �s���Ȃ�-)�B
	// @arg depth_limit
	//     : �őP�肪����ȏ��depth �ł���Β�Ղɓo�^����Ă���őP���Ԃ��B
	// @arg check_depth_limit_last
	//     : ��ԍŌ��depth_limit ���m�F����B
	//           : true �̎�
	//             1. eval_limit �𖞂������]���l�ő�̒��ŁA����depth �ő�̎w�����������B
	//             2. ���̎w�����depth ��depth_limit �𖞂������m�F����B
	//             3. �ȏ�̏����ɍ��v������̂�����΂�����őP��Ƃ��ĕԂ�
	//             (��ԍŌ�Ɉ�x�����`�F�b�N)
	//           : false �̎�
	//             1. eval_limit ��depth_limit �𖞂������]���l�ő�̒��ŁA����depth �ő�̎w�����������B
	//             2. �ȏ�̏����ɍ��v������̂�����΂�����őP��Ƃ��ĕԂ�
	//             (��������^�C�~���O�ŔC�ӂ̎w����ɂ����ă`�F�b�N)
	//             (���炭��˂��牤�̓R�b�`)
	// @return
	//     : first
	//           : �����ɍ����w���肪���������ۂ��B
	//             �����true �ŁAsecond �̎w������w���΂悢�B
	//     : second
	//           : first==true �Ȃ�����ɍ������w����B
	//             first==false �Ȃ�YoBookMove::moveNone()�B
	std::pair<bool, yo_book_move_t> get_best_eval_move(int eval_limit = INT_MIN, int depth_limit = 0, bool check_depth_limit_last = true) const {
		static constexpr int INIT_BESTMOVE_IDX = -1;
		int bestmove_idx = INIT_BESTMOVE_IDX;
		int bestmove_eval = INT_MIN;
		int bestmove_depth = INT_MIN;

		// eval �ő�̎�̒��ŁA�X��depth �ő�̎w�����T���B
		const int moves_size = moves.size();
		for (int i = 0; i < moves_size; ++i) {
			const auto& book_move = moves[i];
			if (book_move.eval < eval_limit) {    // �]���l��������������(������)
				continue;
			}
			if (!check_depth_limit_last) {
				if (book_move.depth < depth_limit) {    // �őP���depth ������������(������)
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

		// ��ȏ�̕]���l �̎w���肪�����������݂��Ȃ����� || �őP���depth ������������(������)
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

		// ��˂��牤�Ŋm��sort ����Ă���͂��ŁA������ł�sort ���Ȃ��B
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

// HACK: ����A���̂Ɍp���ɂ����H���ʂɃ����o�Ɏ�������_��������񂩁H���傢�����Ȃ����H
// ��˂��牤�`���̒�Ղ��A��̋ǖʂɑ΂��Ď����Ă�����
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

// go �R�}���h��parse �̌��ʂ�ێ�
// http://shogidokoro.starfree.jp/usi.html
typedef struct GoInfo {
private:

public:
	static constexpr int TIME_NONE = -1;

	// TODO: ������Aprivate �Ɉڂ���getter, setter ����̃A�N�Z�X�ɂ������B
	bool is_infinite;
	bool is_ponder;    // ���̏ꍇ���Ago inf �Ɠ��l�ŁA�����Ɖ��肵�Ďv�l����B�����AGUI �����btime, wtime �����^������͗l�B
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
	
	// parse ����O�ɖ��x�ĂԂׂ�
	void reset() {
		is_infinite = false;
		is_ponder = false;

        // ���Ԃ̏����l��0 �Ƃ���B
		std::fill_n(time, ColorNum, 0);
		std::fill_n(inc, ColorNum, 0);
		byoyomi = 0;
	}

	// --------------------------------------------------
	// �ȉ��͑S�āA���S��parse ���������Ă���Ăяo������
	// --------------------------------------------------

	inline bool is_inc() const {
		return (inc[Black] > 0 && inc[White] > 0);
	}

	inline bool is_byoyomi() const {
		return (byoyomi > 0);
	}

	// HACK: �������S�Ɏ������Ԃ����� ���Ď��Ⴆ���Ⴄ���O�ȋC������B
	//       is_additional_time_zero �Ƃ��ɕς��悤(������Ȃ񂩂��Ⴄ�B�Ȃ񂩂������O�l���悤���Z����B)�B
	// (http://shogidokoro.starfree.jp/usi.html
	// ��������"�b�ǂ݂Ɖ��Z�������Ƃ�0�Ȃ�byoyomi 0�𑗂�܂��B")
	// @return: �؂ꕉ�������ł��邩�ۂ�
	inline bool is_extra_time_zero(const Color& c) const {
		return (inc[c] == 0 && byoyomi == 0);
	}

	inline int get_time(const Color& c) const { return time[c]; }
	inline int get_inc(const Color& c) const { return inc[c]; }

	// �i�[�������ʂɖ��������������m�F����B
	// true ��Ԃ����Ƃ��A�������������Ƃ̕K�v�����B
	bool check_state() const {
		// ���Z �� �b�ǂ� �݂͌��ɔr���I�ȃI�v�V�����ł���A�������Ă͂Ȃ�Ȃ�
		// (go inf �̏ꍇ���Ago inf �Ɖ��Z ���͕b�ǂ� �͔r���I �Ȃ̂ŕs�K�B)
		if (is_inc() && is_byoyomi()) {
			return false;
		}

		// �v�l���Ԑ������� �Ǝv�l���Ԗ��� �݂͌��ɔr���I�ȃI�v�V�����ł���A�������Ă͂Ȃ�Ȃ��B
		// go infinite �Ȃ�A���Ȃ��Ƃ������l0 ����ω����Ă��Ȃ��͂�(0 ��set ���ꂽ�\���͖ܘ_���邪�l���Ȃ�)�B
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


	HuffmanCodedPosAndEval3 hcpe3;    // �J�n�ǖ�
	std::vector<std::vector<MoveVisits>> candidates_list;    // ��Ǖ��̏�񂪓���B
	std::vector<MoveInfo> move_info_list;                    // ��Ǖ��̏�񂪓���
}reading_hcpe3_task_t;

// HACK
//     : _t �ɁB
//     : ����Apair �̂Ƃ���A���΂�struct ��`���Ă��ǂ��������Ȃ����H����ᕪ����Â炢��B�B�B
// puct player �Ŏg�p�B
#if defined(FEATURE_V2)
    #if defined(USE_PACKED_FEATURE)
        using feature_t = std::tuple<char*, char*>;
		using unpacked_feature_t = std::tuple<float*, float*>;    // nn_tensorrt.hpp �Ŏg�p�B
#else
        using feature_t = std::tuple<float*, float*>;
    #endif
#else
    using feature_t = float*;
#endif