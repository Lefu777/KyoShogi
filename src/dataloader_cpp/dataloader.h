#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <unordered_map>
#include <map>


// https://acceliv.com/2020/06/windows-h-include-error
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef NOMINMAX 
#include <stdio.h>
#include <sys/stat.h>

#include "cshogi.h"
#include "types.hpp"
#include "util.hpp"
#include "stop_watch.hpp"
#include "config.hpp"

// TODO
//     : dlshogi 互換の入力特徴量に対応

// HACK
//     : store 系の関数とかでいちいちローカル変数の__Board board をコンストラクトしてるのは、速度に影響しそうなので、
//       以下のようにグローバル変数一つを永遠に使い続けたい。
//       ただ、ヘッダーに書くのはよろしくないので.cpp に書くべきだが、一部の関数だけ.cpp に行くのが地味気持ち悪いというかなんというか。
//       ある程度一連の流れで使うような関数たちは、同じファイルにとどめておきたい。
//       まぁ、謎なこだわりですな。これは。普通にこのこだわりは割とショーもない気がする。まぁ、気が向いたらグローバル変数使うようにする。
//     : class 自体をtemplate した方がええなこれは。
// dataloader専用
//__Board board_dataloader;

// HACK
//     : read_files_sequential() 関連、
//       _teacher の_next_idx がばらばらに管理されていてキモイ。
//       二つをpair で纏めるとかした方が良さげ。
//       あとは、struct にして、各操作をメンバ関数化するとかね。
//     : 後あれだなb、__FastDataloader<teacher_t>と__FastDataloader<TrainingData> とかにした方がええかもな。
//       後は、程度に継承するとかさ。
//     : store_teacher()って名前、store_data() みたいな汎用性のある名前に変えた方が良い気もする。

// NOTE
//     : https://stackoverflow.com/questions/3052579/explicit-specialization-in-non-namespace-scope
//       class のメンバ関数のtemplate

constexpr unsigned long long hcpe_data_reserve_size = 1 << 27;         // 134217728
constexpr unsigned long long hcpe3_data_reserve_size = 1 << 27;        // 134217728
constexpr int hcpe_TrainingData_candidates_reserve_size = 1 << 2;      // 4
constexpr int hcpe3_TrainingData_candidates_reserve_size = 1 << 6;     // 64

class __FastDataloader {
private:
	//int _i;
	int _batch_size;
	std::string _dir;
	bool _is_dir_set;

	// .bin を読み込む用。
	std::vector<teacher_t> _teachers;

	// hcpe, hcpe3 を読み込む用
	std::vector<TrainingData> _training_data;
	// 重複チェック用 局面に対応するtrainingDataのインデックスを保持
	std::unordered_map<HuffmanCodedPos, unsigned int> _duplicates;

	bool _shuffle;
	std::vector<int> _idxs;    // _teachers or _training_data へのインデックスが0 .. size-1 まで格納されている。(size : _teachers.size())
	                           // shuffle が有効ならrandom に並べ替えられていて、有効でないならarange(size)

	// 既に読み込んだか否かのフラグ付き、ファイルリスト
	// .second == true なら、読み込み済み。
	std::vector<std::pair<std::string, bool>> _file_names_with_flag;
	int _next_idx;    // _teachers の内、既に読み込んだ教師数。
	                  // _teachers の次に読み込むべき教師の一番初めのidx。
             	      // store_teachers_with_sequential_reader() で更新される。
	int _minimum_read_threshold;    // 教師を読み込むときは、少なくともこれ以上の数を読み込む。

public:
	// ※const ObjTy& obj は、左辺値参照、右辺値参照 ともに取れる。
	// @arg dir
	//     : 教師ファイルが入ってるディレクトリ
	//     : ex> teacher というフォルダにある時、dir = "teacher"
	__FastDataloader()
		: _dir(""), _is_dir_set(false), _batch_size(-1), _next_idx(0), _minimum_read_threshold(0), _shuffle(false)
	{
	}

	__FastDataloader(const std::string& dir)
		:_dir(dir), _is_dir_set(true), _batch_size(-1), _next_idx(0), _minimum_read_threshold(0), _shuffle(false)
	{

	}

	__FastDataloader(const std::string& dir, const int& batch_size)
		:_dir(dir), _is_dir_set(true), _batch_size(batch_size), _next_idx(0), 
		_minimum_read_threshold(0), _shuffle(false)
	{

	}

	// sequential reader を用いる時はこっち。
	__FastDataloader(
		const std::string& dir, const int& batch_size, const int& minimum_read_threshold, const bool& shuffle
	)    :_dir(dir), _is_dir_set(true), _batch_size(batch_size),
		_next_idx(0), _minimum_read_threshold(minimum_read_threshold), _shuffle(shuffle)
	{
		if (_minimum_read_threshold < batch_size) {    // 違法
			std::cout
				<< "Error: expected _minimum_read_threshold > batch_size, but "
				<< "_minimum_read_threshold = " << _minimum_read_threshold
				<< ", batch_size = " << batch_size << std::endl << std::flush;
			exit(1);
		}
	}

	//~__FastDataloader() {
	//	finalize();
	//}

	void finalize() {
		std::vector<std::vector<teacher_t>> teacher_auxiliary;
		teacher_auxiliary.push_back({});
		std::swap(teacher_auxiliary.back(), _teachers);

		std::vector<std::vector<TrainingData>> training_data_auxiliary;
		training_data_auxiliary.push_back({});
		std::swap(training_data_auxiliary.back(), _training_data);

		std::vector<std::vector<int>> idxs_auxiliary;
		idxs_auxiliary.push_back({});
		std::swap(idxs_auxiliary.back(), _idxs);

		_duplicates.clear();
	}

	// 教師が格納されているvector への参照を返す。
	template<typename T>
	auto& get_data() {
		if constexpr (std::is_same_v<T, teacher_t>) {
			return _teachers;
		}
		else if constexpr (std::is_same_v<T, TrainingData>) {
			return _training_data;
		}
		else {
			throw std::runtime_error("get_data<>() got unsupported typename.");
		}
	}

	// ファイルの拡張子が適格か否か
	template<typename T>
	static bool file_extension_is_ok(const std::string& filename) {
		if constexpr (std::is_same_v<T, teacher_t>) {
			return filename.substr(filename.size() - 4) == ".bin";
		}
		else if constexpr (std::is_same_v<T, TrainingData>) {
			return (
				filename.substr(filename.size() - 5) == ".hcpe"
				|| filename.substr(filename.size() - 6) == ".hcpe3"
			);
		}
		else {
			throw std::runtime_error("file_extension_is_ok<>() got unsupported typename.");
		}
	}

	void set_batch_size(const int& batch_size) { _batch_size = batch_size; }

	void set_dir(const std::string& dir) {
		_dir = dir;
		_is_dir_set = true;
	}
	bool is_dir_set() const { return _is_dir_set; }

	void print_teachers() const;
	static void print_teachers(const std::vector<teacher_t>& teachers);


	// NOTE
	//     : これ、恐らくもう使わないし、使うべきでない。
	// file_names をメンバ変数として保持しておいて、個の関数内でデータ数が不足すれば、その都度読み込む感じにしよう。最終的には。
	// TODO: cython に定義部分コピペして、ラップ。python で動くかtest.
	[[deprecated("This is because it can only store the input features out of the data required for training. Use other store_features() instead.")]]
	void store_features(char* _mem, const int mem_size, const int start_idx) const {
		std::cout << "Error: __FastDataloader::store_features() is deprecated." << std::endl;
		exit(1);


		//float* mem = reinterpret_cast<float*>(_mem);
		//if (mem_size % N_FEATURE_WHC != 0) {
		//	std::cout << "Error: memory size is unsuitable, mem_size = " << mem_size << std::endl << std::flush;
		//	exit(1);
		//}

		//const int batch_size = mem_size / N_FEATURE_WHC;
		//const int teachers_size = _teachers.size();
		//__Board board;

		//for (int i = 0; i < batch_size; ++i) {
		//	board.set(_teachers[(start_idx + i) % teachers_size].sfen);
		//	board.make_input_features(mem + (N_FEATURE_WHC * i));
		//}
	}

	//////// 一つの教師データを、i番目のbatchにstore する関数。
	// @arg _nd_xxxx: nd_labels / nd_probabilitys
	// @arg i: batch の番号。何番目のbatch か。
	template<typename T>
	void _store_one_teacher(
		char* _ndfeatures,
		char* _nd_xxxx,
		char* _ndvalues,
		char* _ndresults,
		__Board& board,
		const T& data,
		const int& i
	) const {
		if (std::is_same_v<T, teacher_t>) {
			float* features = reinterpret_cast<float*>(_ndfeatures);
			int64_t* labels = reinterpret_cast<int64_t*>(_nd_xxxx);
			float* values = reinterpret_cast<float*>(_ndvalues);
			float* results = reinterpret_cast<float*>(_ndresults);

			board.set(data.sfen);
			board.make_input_features(features + (N_FEATURE_WHC * i));

			labels[i] = move_to_label(Move(data.move), board.turn());

			values[i] = data.value;
			results[i] = data.result;
		}
		else if (std::is_same_v<T, TrainingData>) {
			float* features = reinterpret_cast<float*>(_ndfeatures);
			float* probabilitys = reinterpret_cast<float*>(_nd_xxxx);
			float* values = reinterpret_cast<float*>(_ndvalues);
			float* results = reinterpret_cast<float*>(_ndresults);

			board.set_hcp(data.hcp);
			board.make_input_features(features + (N_FEATURE_WHC * i));

			const int& offset = N_LABEL_SIZE * i;
			for (const auto& kv : data.candidates) {
				//const int& label = __dlshogi_make_move_label(kv.first, board.turn());
				const int& label = move_to_label(Move(kv.first), board.turn());
				probabilitys[offset + label] = kv.second;
			}

			values[i] = score_to_winrate(data.eval / data.count, PONANZA);
			results[i] = data.result / data.count;
		}
		else {
			throw std::runtime_error("Error: _store_one_teacher<>() got unexpected typename.");
		}
	}

	//////// ある地点からある地点までの教師データを numpy配列にstore
	// TODO
	//     : 例えば、teacher.size() を超えてしまった場合は、0 を返して、
	//       python 側では0 を受け取ったら、data からもうbatch_size個取り出すことは出来ないとして、
	//       これ以上mini_batch() を呼ばない、的な処理をする。
	// @arg _nd_xxxx: nd_labels / nd_probabilitys
	template<typename T>
	void _store_teachers(
		char* _ndfeatures,
		char* _nd_xxxx,
		char* _ndvalues,
		char* _ndresults,
		const int batch_size,
		const int start_idx    // 開始地点までのoffset とも言える。
	) const {
		if (std::is_same_v<T, teacher_t>) {
			if (batch_size <= 0) {
				std::cout << "Error: [__FastDataloader::_store_teachers<>()] batch_size = " << batch_size << " <= 0" << std::endl;
				exit(1);
			}

			__Board board;
			const int& teachers_size = _teachers.size();

			for (int i = 0; i < batch_size; ++i) {
				const teacher_t& teacher = _teachers[(start_idx + i) % teachers_size];
				_store_one_teacher<teacher_t>(_ndfeatures, _nd_xxxx, _ndvalues, _ndresults, board, teacher, i);
			}
		}
		else if (std::is_same_v<T, TrainingData>) {
			if (batch_size <= 0) {
				std::cout << "Error: [__FastDataloader::_store_teachers<>()] idxs.size() = " << batch_size << " <= 0" << std::endl;
				exit(1);
			}

			__Board board;
			const int& data_size = _training_data.size();

			float* probabilitys = reinterpret_cast<float*>(_nd_xxxx);
			std::fill_n(probabilitys, N_LABEL_SIZE * batch_size, 0);

			for (int i = 0; i < batch_size; ++i) {
				const TrainingData& training_data = _training_data[(start_idx + i) % data_size];

				_store_one_teacher<TrainingData>(_ndfeatures, _nd_xxxx, _ndvalues, _ndresults, board, training_data, i);
			}
		}
		else {
			throw std::runtime_error("Error: _store_teachers<>() got unexpected typename.");
		}
	}

	// HACK
	//     : ちょっとくどい気もする。デフォルト引数とかでも対応できるのでは？知らんけど。
	//        -> てかこれ意味あるんか？昔の私は何を考えてこんなゴミを書いたんや。。。？
	// _store_teachers() のラッパー。
	// batch_size = _batch_size, とする
	template<typename T>
	void store_teachers(
		char* _ndfeatures,
		char* _nd_xxxx,
		char* _ndvalues,
		char* _ndresults,
		const int& start_idx
	) const {
		_store_teachers<T>(_ndfeatures, _nd_xxxx, _ndvalues, _ndresults, _batch_size, start_idx);
	}

	//////// vectorによるインデックス一覧で指定された教師データを numpy配列にstore
	// @arg idxs: storeしたい教師のインデックス のリスト
	template<typename T>
	void store_teachers(
		char* _ndfeatures,
		char* _nd_xxxx,
		char* _ndvalues,
		char* _ndresults,
		const std::vector<int>& idxs
	) const {
		if (std::is_same_v<T, teacher_t>) {
			if (idxs.size() <= 0) {
				std::cout << "Error: [__FastDataloader::store_teachers()] idxs.size() = "
					<< idxs.size() << " <= 0" << std::endl;
				exit(1);
			}

			__Board board;
			const int& batch_size = idxs.size();

			for (int i = 0; i < batch_size; ++i) {
				const int& teacher_idx = idxs[i];
				const teacher_t& teacher = _teachers[teacher_idx];

				_store_one_teacher<teacher_t>(_ndfeatures, _nd_xxxx, _ndvalues, _ndresults, board, teacher, i);
			}
		}
		else if (std::is_same_v<T, TrainingData>) {
			if (idxs.size() <= 0) {
				std::cout << "Error: [__FastDataloader::store_teachers()] idxs.size() = "
					<< idxs.size() << " <= 0" << std::endl;
				exit(1);
			}

			__Board board;
			const int& batch_size = idxs.size();

			float* probabilitys = reinterpret_cast<float*>(_nd_xxxx);
			std::fill_n(probabilitys, N_LABEL_SIZE * batch_size, 0);

			for (int i = 0; i < batch_size; ++i) {
				const int& teacher_idx = idxs[i];
				const TrainingData& training_data = _training_data[teacher_idx];

				_store_one_teacher<TrainingData>(_ndfeatures, _nd_xxxx, _ndvalues, _ndresults, board, training_data, i);
			}
		}
		else {
			throw std::runtime_error("store_teachers<>() got unexpected typename.");
		}
	}

	// for python
	template<typename T>
	void store_teachers_with_idxs(
		char* _ndfeatures,
		char* _nd_xxxx,
		char* _ndvalues,
		char* _ndresults,
		const std::vector<int>& idxs
	) const {
		store_teachers<T>(_ndfeatures, _nd_xxxx, _ndvalues, _ndresults, idxs);
	}

	//////// reader
	static int _get_file_list(std::vector<std::string>& file_names, std::string dir);
	static int _read_one_file(std::vector<teacher_t>& teachers, std::string file_path);
	static int _read_one_file(
		std::vector<TrainingData>& training_data,
		std::unordered_map<HuffmanCodedPos, unsigned int>& duplicates,
		std::string file_path
    );
	template<typename T>
	int _read_one_file(const std::string& file_path) {
		if constexpr (std::is_same_v<T, teacher_t>) {
			return _read_one_file(_teachers, file_path);
		}
		else if constexpr (std::is_same_v<T, TrainingData>) {
			return _read_one_file(_training_data, _duplicates, file_path);
		}
		else {
			throw std::runtime_error("Error: _read_one_file<>() got unexpected typename.");
		}
	}
	static int _read_files_all(
		std::vector<teacher_t>& teachers,
		std::vector<TrainingData>& training_data,
		std::unordered_map<HuffmanCodedPos, unsigned int>& duplicates,
		std::string dir
	);
	int read_files_all();

	// NOTE
	//     : sequential reader を用いている際は、今_teachers に入ってる教師数が返る。
	//     : 一括で読み込む場合は、合計教師数が変える。

	template<typename T> int size() {
		return get_data<T>().size();
	}

	// 排他的なsize()
	// __FastDataloader は、二つの教師タイプを同時に読み込めないので、
	// 片方のみが0超過である時のみsize() を返し、それ以外では0 を返す。
	int xor_size() {
		if (size<teacher_t>() == 0 && size<TrainingData>() == 0) {
			return 0;
		}
		else if (size<teacher_t>() > 0 && size<TrainingData>() == 0) {
			return size<teacher_t>();
		}
		else if (size<teacher_t>() == 0 && size<TrainingData>() > 0) {
			return size<TrainingData>();
		}
		else {
			return 0;
		}
	}

	//////// shuffle
	static int shuffle_files(std::string src_dir, std::string dst_dir, std::string base_file_name);
	static int shuffle_write_one_file(std::string file_path, std::string dst_dir, std::string base_file_name);

	//////// sequential reader 関連
	// まだ使ってない末尾にある残り物のデータ以外の、全てをクリアして、
	// 末尾にあった未使用のデータを先頭に持ってくる。
	template<typename T>
	void __clear_data_except_unused(const int& unused_start_idx) {
		if constexpr (!(std::is_same_v<T, teacher_t> || std::is_same_v<T, TrainingData>)) {
			//std::cout << "Error: unexpected typename" << std::endl;
			throw std::runtime_error("Error: __clear_data_except_unused<>() got unexpected typename.");
		}
		//std::cout << "info: get_data() start" << std::endl;
		std::vector<T>& data = get_data<T>();

		stopwatch_t sw;
		stopwatch_t sw_auxiliary;

		//std::cout << "[__clear_data_except_unused()] start" << std::endl;
		//sw_auxiliary.start_with_print("    if (data.size() > 0))");
		if (data.size() > 0) {
			//sw.start_with_print("    copy unused data to \"std::vector<T> data_unused\"");
			// インデックスに従って、未使用のデータを先頭に持ってくる。
			const int idxs_size = _idxs.size();
			std::vector<T> data_unused;
			for (int i = unused_start_idx; i < idxs_size; ++i) {
				const int& idx = _idxs[i];
				data_unused.emplace_back(data[idx]);
			}
			//sw.stop_with_print();

			// TODO: 一応、clear() した場合の速度も確認する
			// https://codeforces.com/blog/entry/118138
			//sw.start_with_print("    std::swap(auxiliary.back(), data)");
			// 2倍ぐらい速くなった。
			std::vector<std::vector<T>> auxiliary;    // if のスコープを抜けるときに解放される。
			auxiliary.push_back({});
			//std::cout << "[    data.size() = " << data.size() << "]" << std::endl;
			//std::cout << "[    data.capacity() = " << data.capacity() << "]" << std::endl;
			std::swap(auxiliary.back(), data);
			//std::cout << "[    data.capacity() = " << data.capacity() << "]" << std::endl;
			//sw.stop_with_print();

			//sw.start_with_print("    data = std::move(data_unused)");
			data = std::move(data_unused);
			//std::cout << "[    data.size() = " << data.size() << "]" << std::endl;
			//sw.stop_with_print();

			//sw.start_with_print("    data.shrink_to_fit()");
			//data.shrink_to_fit();
			//sw.stop_with_print();

			//sw.start_with_print("    _idxs.clear()");
			_idxs.clear();
			//sw.stop_with_print();

			//sw.start_with_print("    _idxs.shrink_to_fit()");
			//_idxs.shrink_to_fit();
			//sw.stop_with_print();
		}
		//sw_auxiliary.stop_with_print();

		if constexpr (std::is_same_v<T, TrainingData>) {
			if (_training_data.size() > 0) {
				// 重複の辞書を作成しなおす

				//sw.start_with_print("    _duplicates.clear()");
				_duplicates.clear();
				//sw.stop_with_print();


				//sw.start_with_print("    _duplicates.swap()");
				//std::unordered_map<HuffmanCodedPos, unsigned int> tmp_dupl;
				//std::swap(tmp_dupl, _duplicates);
				//sw.stop_with_print();

				//sw.start_with_print("    _duplicates.reserve(data_reserve_size)");
				// HACK: これ、data_reseve_size にしようね。
				_duplicates.reserve(hcpe3_data_reserve_size);
				//sw.stop_with_print();


				//sw.start_with_print("remake _duplicates");
				const unsigned int size = _training_data.size();
				for (unsigned int i = 0; i < size; ++i) {
					_duplicates.emplace(_training_data[i].hcp, i);
				}
				//sw.stop_with_print();
			}
		}
		//std::cout << "[__clear_data_except_unused()] done" << std::endl;
	}

	// read_files_sequential() を用いる時に、loop の開始前に呼ぶ。
	// つまり、__iter__() で呼ぶべし。
	template<typename T>
	void init_at_iter() {
		for (auto& ff : _file_names_with_flag) {
			ff.second = false;
		}

		__clear_data_except_unused<T>(_next_idx);
		_next_idx = 0;

		// NOTE
		//     : sequential reader の時には、_file_names_with_flag.size() > 0となる。
		//     -> sequential reader 且つshuffle が有効なら、読み込むファイルの順番をシャッフルする。
		if (_shuffle && _file_names_with_flag.size() > 0) {
			shuffle(_file_names_with_flag);
		}
	}

	template<typename T>
	int __read_files_sequential(const int unused_start_idx, const int read_threshold);

	// _read_files_sequential() のラッパー
	// 内部的に更新している_next_idxと、予め保持している_minimum_read_threshold を引数に与えることで、
	// ユーザー側(私@python)はよだれ垂らしながらでも使える。
	template<typename T>
	int read_files_sequential() {
		return __read_files_sequential<T>(_next_idx, _minimum_read_threshold);
	}

	// _store_teachers() のラッパー。
	// batch_size = _batch_size, start_idx = _next_idx とする
	// 内部で_next_idx を保持して更新しているので、ユーザー側(私@python)はよだれ垂らしながらでも使える。
	template<typename T>
	int store_teachers_with_sequential_reader(
		char* _ndfeatures,
		char* _nd_xxxx,    // _nd_move_labels or _ndprobabilitys
		char* _ndvalues,
		char* _ndresults
	) {
#ifdef DEBUG_
		std::cout << "info: _teachers.size() = " << get_data<T>().size() << ", _next_idx = " << _next_idx << std::endl;
#endif
		const auto& data = get_data<T>();

		// NOTE
		//     : _teacher からギリギリまでbatch を搾り取らないままにinit_at_iter() が走ってしまうと、
		//       _next_idx = 0 となってしまい、昔に読み込んだデータをもう一度batch として渡し始めてしまう。
		//       実質的に_teacher をクリアしているread_files_sequential が呼ばれない為である。
		const int n_extra_data = data.size() - _next_idx;
		if (n_extra_data < _batch_size) {    // 残りデータが_batch_size より少ない。
			//std::cout << "info: not enough teacher, so read teacher" << std::endl;
			if (!read_files_sequential<T>()) {
				std::cout << "Error: Not enough data is available." << std::endl;
				return 0;
			}
		}
		std::vector<int> idxs_tmp;
		std::copy(_idxs.begin() + _next_idx, _idxs.begin() + _next_idx + _batch_size, std::back_inserter(idxs_tmp));

		store_teachers<T>(_ndfeatures, _nd_xxxx, _ndvalues, _ndresults, idxs_tmp);
		_next_idx += _batch_size;
		return 1;
	}

	// まだ学習に用いていないデータが残っているか
	// @return
	//     : 残っているならtrue
	//       残っていないならfalse
	template<typename T>
	bool have_extra_batch() {
		if (_batch_size <= 0) {
			std::cout << "Error: [__FastDataloader::have_extra_batch()] _batch_size = " << _batch_size << " <= 0" << std::endl;
			exit(1);
		}

		// まだファイル一覧さえも無いのであれば、確実に読み込めるbatch があるはず。
		if (_file_names_with_flag.size() <= 0) {
			return true;
		}

		const auto& last_ff = _file_names_with_flag.back();
		if (last_ff.second) {    // 最後のファイルを読み込み済み
			const int n_extra_data = get_data<T>().size() - _next_idx;
			if (n_extra_data < _batch_size) {    // 残りデータが_batch_size より少ない。
				return false;
			}
		}
		return true;
	}

	//////// sequential reader 関連 (TrainingData のみに対応)

//	// _store_teachers() のラッパー。
//	// batch_size = _batch_size, start_idx = _next_idx とする
//	// 内部で_next_idx を保持して更新しているので、ユーザー側(私@python)はよだれ垂らしながらでも使える。
//	int store_teachers_random_with_sequential_reader(
//		char* _ndfeatures,
//		char* _ndmove_labels,
//		char* _ndvalues,
//		char* _ndresults
//	) {
//#ifdef DEBUG_
//		std::cout << "info: _teachers.size() = " << _teachers.size() << ", _next_idx = " << _next_idx << std::endl;
//#endif
//		// NOTE
//		//     : _teacher からギリギリまでbatch を搾り取らないままにinit_at_iter() が走ってしまうと、
//		//       _next_idx = 0 となってしまい、昔に読み込んだデータをもう一度batch として渡し始めてしまう。
//		//       実質的に_teacher をクリアしているread_files_sequential が呼ばれない為である。
//		const int n_extra_data = _teachers.size() - _next_idx;
//		if (n_extra_data < _batch_size) {    // 残りデータが_batch_size より少ない。
//			//std::cout << "info: not enough teacher, so read teacher" << std::endl;
//			if (!read_files_sequential<TrainingData>()) {
//				std::cout << "Error: Not enough data is available." << std::endl;
//				return 0;
//			}
//		}
//		_store_teachers(_ndfeatures, _ndmove_labels, _ndvalues, _ndresults, _batch_size, _next_idx);
//		_next_idx += _batch_size;
//		return 1;
//	}
};
