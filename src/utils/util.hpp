#pragma once

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <atomic>
#include "types.hpp"
#include "MT.hpp"

// TODO
//     : �߂�����K���₩�炿����Ɛ����������ˁB

float my_round(float x, int d);

template<typename T>
inline std::vector<T> arange(int start, int size, int step);
template<typename T>
inline std::vector<T> arange(int start, int end);
template<typename T>
inline std::vector<T> arange(int size);

class Progress {
private:
	std::string _process_name;
	int _i;
	int _total;
	int _interval;

	void _print(int curr) {
		std::cout << "\r" << _process_name << " : "
			<< std::fixed << std::setprecision(2) << (100.0 * curr) / _total
			<< " % [" << curr << " / " << _total << "]";
	}

public:
	// @arg total: max step���B
	// @arg interval: ��step ���Ƃɕ\�����X�V���邩�B
	Progress(std::string process_name, int total, int interval)
		: _process_name(process_name), _interval(interval), _total(total)
	{
		_i = 0;
		_print(0);
	}

	// ��step ���ƂɌĂ�ŁA�C���N�������g
	void step() {
		++_i;
		if (_i % _interval == 0) {
			_print(_i);
		}
	}

	// �Ōオ�Y��Ɋ���؂�鐔���ŏI���Ƃ͌���Ȃ��̂ŁA100% ���Ō�ɕ\������B
	void finalize() {
		_print(_total);
		std::cout << std::endl;
	}
};

std::string get_now_str_for_filename();
std::string get_now_str();

// NOTE
//     : �f�t�H���g�����ݒ肷��ƂȂ񂩃o�O��B
void write_teacher(
	const std::vector<teacher_t>& teachers,
	std::string base_file_name,
	int output_interval,
	bool random_name,
	bool verbose
);



std::vector<std::string> tokenize(std::string cmd);

// �n��seq�̒��Ńg�[�N��tok���ŏ��Ɍ��������C���f�b�N�X��Ԃ��B������Ȃ������ꍇ��-1��Ԃ��B(by bingAI)
// @arg seq
//     : �󔒂ɂ��token ���Ƃɕ����邱�Ƃ��o����n��
// @arg query_token
//     : �n��Ɋ܂܂�邩�ۂ���m�肽��token(�T������token)
// @return
//     : �n��Ɏw�肳�ꂽtoken ���܂܂�邩�ۂ��B
//     : ex>
//       seq = "position startpos moves 2g2f"
//       query_token = "moves"
//       -> 2
//     : ex>
//       seq = "position startpos"
//       query_token = "moves"
//       -> -1
inline int index_of_token(const std::string seq, const std::string& query_token) {
	auto tokens = tokenize(seq);
	for (const auto& token : tokens) {
		if (token == query_token) {
			return (&token - &tokens[0]);
		}
	}
	return -1;
}

inline bool find_token(const std::string seq, const std::string& query_token) {
	// HACK: -1�A�}�W�b�N�i���o�[�Ȃ񂾂�Ȃ��`
	return (index_of_token(seq, query_token) != -1);
}


// https://qiita.com/i153/items/38f9688a9c80b2cb7da7
//template<class T> int argmax_idx(const T& array, int size);
// �ő�l������idx ��Ԃ��B
template<class T>
inline int argmax_idx(const T& array, int size) {
	auto crrt_max_element = array[0];
	int crrt_max_idx = 0;
	for (int i = 1; i < size; ++i) {
		if (array[i] > crrt_max_element) {
			crrt_max_element = array[i];
			crrt_max_idx = i;
		}
	}
	return crrt_max_idx;
}

template<class T>
inline void shuffle(std::vector<T>& vec) {
	int target;
	int size = vec.size();
	T tmp;

	//�Ō�����珇�����[�v����
	for (int i = size - 1; i > 0; --i) {
		//0~i�Ԗڂ̗v�f����A����ւ��Ώۂ̗v�f�������_���ɑI��
		// unsigned long randnum = genrand_int32();
		// printf("debug: randnum = %lu\n", randnum);
		target = mt_genrand_int32() % i;

		//�������牺�͖����̗v�f�Ɠ���ւ�
		tmp = vec[target];
		vec[target] = vec[i];
		vec[i] = tmp;
	}
}

template<typename T>
inline std::vector<T> arange(int start, int size, int step) {
	std::vector<T> vec;
	for (int i = start; i < start + size; i += step) {
		vec.emplace_back(i);
	}
	return vec;
}

// start�ȏ�Aend�����̐����̔z���Ԃ�
template<typename T>
inline std::vector<T> arange(int start, int end) {
	return arange<T>(start, end, 1);
}

// 0�ȏ�Asize�����́A�v�f����size�̐����̔z���Ԃ�
template<typename T>
inline std::vector<T> arange(int size) {
	return arange<T>(0, size);
}

// �W���t���V�O���C�h
// �֋X��A-x�̌W���̋t����scale �Ɠǂ�ł���B
inline float sigmoid_with_scale(const float& x, const float& scale) {
	return 1 / (1 + expf(-x / scale));
}

// �ʏ�̃V�O���C�h
inline float sigmoid(const float& x) {
	return sigmoid_with_scale(x, 1);
}

// NOTE: ponanza�萔���������ƁA�����]���l�ł���菟���������o��B
// �]���l���珟��
inline float score_to_winrate(const float& score, const float& ponanza) {
	return sigmoid_with_scale(score, ponanza);
}

// python-dlshogi2 �̎��ƁA�`�����Ⴆ�ǁA�l�͈�v�B
// ��������]���l
inline float winrate_to_score(const float& winrate, const float& ponanza) {
	return ponanza * logf(winrate / (1 - winrate));
}

// ��ԍŌ�̍s�����s���Ă��邩���Ȃ����Ɋւ�炸�A���s���[�ǂ��t�����Ƃɒ��ӂ��Ďg���ׂ��B
// @arg file_path: �ǂݍ��݃t�@�C���̃p�X
// @return
//     : �ǂݍ��񂾃t�@�C���̓��e�����string �ɋl�ߍ��񂾕��B
inline std::string get_file_str(const std::string& file_path) {
	std::ifstream reading_file;
	reading_file.open(file_path, std::ios::in);
	std::string buf;
	std::string file_str;
	while (std::getline(reading_file, buf)) {
		file_str += buf + "\n";
	}
	return file_str;
}

// �Ȃ񂩖{����iterable ���ۂ��͂����ƐF�X���Ȃ�����炵�����ǁA
// �܂��ǂ���vector ��initializer_list ���炢�����g��񂵁B�B�B
template<typename T>
inline auto join(std::string separator, const T& strs) -> decltype(strs.begin(), strs.end(), strs.size(), std::string()) {
	std::string tmp = "";
	if (strs.size() > 0) {
		const auto strs_end = strs.end();
		for (auto iter = strs.begin(); iter != strs_end - 1; ++iter) {
			tmp += *iter + separator;
		}
		tmp += *(strs_end - 1);
	}
	return tmp;
}

// TODO
//     : ��ݍ��ݎ��g���ď����񂩂Ȃ��`�H
// https://zenn.dev/melos/articles/a7262ffce8126d
// join(" ", "abc", "def", "ghi") -> "abc def ghi"
template<typename... T>
inline std::string join(std::string separator, T... strs) {
	std::initializer_list<std::string> strs_list{ strs... };
	return join(separator, strs_list);
}

// �������A�����_�ȉ���؂�̂Ăĕ�����
inline std::string decimal_to_int_str(const double& pos_deci) {
	return std::to_string(static_cast<long long>(pos_deci));
}

// ���̏������A�����_�ȉ���؂�̂Ăĕ�����
inline std::string positive_decimal_to_int_str(const double& pos_deci) {
	if (pos_deci < 0) {
		std::cout << "Error: failed to ms_to_str()" << std::endl;
		throw std::runtime_error("Error");
	}

	return std::to_string(static_cast<unsigned long long>(pos_deci));
}

template <typename T>
constexpr bool false_v = false;
template <typename T>
constexpr bool true_v = false;

// HACK: �Ȃ�ŁA�W����template�������́H����Ȏ����Ă���H
// https://marycore.jp/prog/cpp/convert-string-to-number/
// T �Ŏw�肵���^��string ��ϊ�
template<typename T>
inline T stox(std::string str) {
	if constexpr (std::is_same_v<T, std::string>) {
		return str;
	}
	else if constexpr (std::is_same_v<T, int>) {
		return std::stoi(str);
	}
	else if constexpr (std::is_same_v<T, long>) {
		return std::stol(str);
	}
	else if constexpr (std::is_same_v<T, long long>) {
		return std::stoll(str);
	}
	else if constexpr (std::is_same_v<T, unsigned long>) {
		return std::stoul(str);
	}
	else if constexpr (std::is_same_v<T, unsigned long long>) {
		return std::stoull(str);
	}
	else if constexpr (std::is_same_v<T, float>) {
		return std::stof(str);
	}
	else if constexpr (std::is_same_v<T, double>) {
		return std::stod(str);
	}
	else {
		static_assert(false_v<T>, "Error: stox() got invalid typename T.");
	}
}

// T �� �C���f�b�N�X ��pair ���AT �Ŕ�r�B
// �~��
template<typename T>
inline bool compare_descending(const std::pair<T, int>& x, const std::pair<T, int>& y) {
	return x.first > y.first;
}

// ����
template<typename T>
inline bool compare_ascending(const std::pair<T, int>& x, const std::pair<T, int>& y) {
	return x.first < y.first;
}

//// T �� �C���f�b�N�X ��pair ���AT �Ŏw�肳�ꂽ�֐���p���Ĕ�r�B
//template<typename T>
//inline bool(*getcustome_compare_func(std::pair<T, int>&, const std::pair<T, int>&)) (
//	bool(*compare_fp)(T, T)
//) {
//	return [&compare_fp](std::pair<T, int>& x, const std::pair<T, int>& y) {
//		return compare_fp(x.first, y.first);
//	};
//}
//


// vec ���~����sort ���āA�~����index ��Ԃ�
// �y���͖����̂ŁA�p�ɂɌĂяo���Ȃ�����
template<typename T>
inline std::vector<int> sorted_idx_descending(const std::vector<T>& vec) {
	std::vector<std::pair<T, int>> tmp;    // �v�f & �C���f�b�N�X ��vector
	std::vector<int> ret;    // idx ��vec
	const int& size = vec.size();
	for (int i = 0; i < size; ++i) {
		tmp.emplace_back(vec[i], i);
	}

	std::sort(tmp.begin(), tmp.end(), compare_descending<T>);

	for (int i = 0; i < size; ++i) {
		ret.emplace_back(tmp[i].second);
	}
	return ret;
}

// TODO
//     : https://zenn.dev/reputeless/books/standard-cpp-for-competitive-programming/viewer/library-algorithm
//       ���N�����K�v�Ȃ�A�ʂ̊֐����g���ׂ��B������B
// �y���͖����̂ŁA�p�ɂɌĂяo���Ȃ�����
// �z�� ���~����sort ���āA�~����index ��Ԃ�
template<typename T>
inline std::vector<int> sorted_idx_descending(const std::unique_ptr<T[]>& array, const int& size) {
	std::vector<std::pair<T, int>> tmp;
	std::vector<int> ret;    // idx ��vec
	for (int i = 0; i < size; ++i) {
		tmp.emplace_back(array[i], i);
	}

	std::sort(tmp.begin(), tmp.end(), compare_descending<T>);

	for (int i = 0; i < size; ++i) {
		ret.emplace_back(tmp[i].second);
	}
	return ret;
}

// https://www.paveway.info/entry/2020/10/17/cli_lamdafunctionarg
// ��r�֐����J�X�^���o�����
// @arg compare_fp: T ���r����֐�
template<typename T, typename F>
inline std::vector<int> sorted_idx(
	const std::unique_ptr<T[]>& array, const int& size, F compare_func) {
	std::vector<std::pair<T, int>> tmp;
	std::vector<int> ret;    // idx ��vec
	for (int i = 0; i < size; ++i) {
		tmp.emplace_back(array[i], i);
	}

	std::sort(tmp.begin(), tmp.end(), [&compare_func](std::pair<T, int>& x, const std::pair<T, int>& y) {
	        return compare_func(x.first, y.first);
	    }
	);

	for (int i = 0; i < size; ++i) {
		ret.emplace_back(tmp[i].second);
	}
	return ret;
}

// thread safe ��IO
enum YoSyncCout { IO_LOCK, IO_UNLOCK };
std::ostream& operator<<(std::ostream&, YoSyncCout);

#define sync_cout std::cout << IO_LOCK
#define sync_endl std::endl << IO_UNLOCK

// bool to str
#define bts(b) ((b) ? "true" : "false")

// min, max
#define my_min(a, b) ((a) < (b) ? (a) : (b))
#define my_max(a, b) (-my_min(-a, -b))

// str to bool
inline bool stb(const std::string& str) {
	if (str == "true") {
		return true;
	}
	else if (str == "false") {
		return false;
	}
	else {
		std::stringstream ss;
		ss << "Error: stb() failed to convert, because got str = [" << str << "]";
		throw std::runtime_error(ss.str());
	}
}

// https://en.cppreference.com/w/cpp/atomic/atomic/fetch_add
// ��interger �n�A�܂�floating �n��fetch_add()�B
// c++20 ���O�ł�integer �ɂ����Ή����ĂȂ��̂ŁA�������őΉ����Ă��
template<typename T>
inline void atomic_fetch_add(std::atomic<T>* obj, T arg) {
	T expected = obj->load();
	while (!obj->compare_exchange_weak(expected, expected + arg)) {

	}
}

template<typename T>
inline void atomic_fetch_sub(std::atomic<T>* obj, T arg) {
	T expected = obj->load();
	while (!obj->compare_exchange_weak(expected, expected - arg)) {

	}
}

// dlshogi �̃I�v�V����������A�������ł̖��O�ɕϊ��B
const std::unordered_map<std::string, std::string> option_name_from_dlshogi = {
	{"C_base", "puct_c_base"},
	{"C_init", "puct_c_init_x100"},
	{"C_fpu_reduction", "puct_c_fpu_reduction_x100"},
	{"C_base_root", "puct_c_base_root"},
	{"C_init_root", "puct_c_init_root_x100"},
	{"C_fpu_reduction_root", "puct_c_fpu_reduction_root_x100"},
	{"Softmax_Temperature", "softmax_temperature_x100"},
};
