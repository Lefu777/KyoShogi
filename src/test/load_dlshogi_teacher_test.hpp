#pragma once
#include <iostream>
#include <fstream>
#include <numeric>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <exception>
#include <map>

#include "cshogi.h"

#include "types.hpp"
#include "util.hpp"
#include "stop_watch.hpp"


constexpr uint64_t __MUTEX_NUM = 65536; // must be 2^n
std::mutex __mutexes[__MUTEX_NUM];
inline std::mutex& __GetPositionMutex(const Position* pos)
{
	// NOTE
	//     : pos->getKey() & (MUTEX_NUM - 1) が簡易的なHash になってる感じ？
	//       それによって、同じ局面を同時に操作しない、的な...?
	return __mutexes[pos->getKey() & (__MUTEX_NUM - 1)];
}

// ====================================================================================================
// dataloader.cpp から持ってきたやつ。名前の先頭にアンダーバーがついてるだけ。
// ====================================================================================================
// make result
inline float __make_result(const uint8_t result, const Color color) {
	const GameResult gameResult = (GameResult)(result & 0x3);
	if (gameResult == Draw)
		return 0.5f;

	if ((color == Black && gameResult == BlackWin) ||
		(color == White && gameResult == WhiteWin)) {
		return 1.0f;
	}
	else {
		return 0.0f;
	}
}

// hcpe形式の指し手をone-hotの方策として読み込む
// 複数回呼ぶことで、複数ファイルの読み込みが可能
// @arg use_average: 重複局面について、平均を取る。
inline size_t __load_hcpe(
	std::vector<TrainingData>& training_data, std::unordered_map<HuffmanCodedPos, unsigned int>& duplicates,
	std::ifstream& ifs, bool use_average, const double eval_scale, size_t& len
) {
	for (int p = 0; ifs; ++p) {
		HuffmanCodedPosAndEval hcpe;
		ifs.read((char*)&hcpe, sizeof(HuffmanCodedPosAndEval));
		if (ifs.eof()) {
			break;
		}

		const int eval = (int)(hcpe.eval * eval_scale);
		if (use_average) {
			auto ret = duplicates.emplace(hcpe.hcp, training_data.size());
			if (ret.second) {
				auto& data = training_data.emplace_back(
					hcpe.hcp,
					eval,
					__make_result(hcpe.gameResult, hcpe.hcp.color())
				);
				data.candidates[hcpe.bestMove16] = 1;
			}
			else {
				// NOTE
				//     : https://cpprefjp.github.io/reference/unordered_map/unordered_map/emplace.html
				//       重複局面で合った場合、過去に登録した時の箇所へのiterator : ret.first
				//       であり、ret.first->second にはkey/value のvalue が入っており、
				//       そのvalue はduplicates の定義より、trainingData のインデックス。
				//       重複データの場合、加算する(hcpe3_decode_with_valueで平均にする)
				auto& data = training_data[ret.first->second];
				data.eval += eval;
				data.result += __make_result(hcpe.gameResult, hcpe.hcp.color());
				data.candidates[hcpe.bestMove16] += 1;    // NOTE: この局面において、hcpe.bestMove16 が指された回数 +1
				// hcpe3 化する時に役立つよね。
				data.count++;
			}
		}
		else {
			auto& data = training_data.emplace_back(
				hcpe.hcp,
				eval,
				__make_result(hcpe.gameResult, hcpe.hcp.color())
			);
			data.candidates[hcpe.bestMove16] = 1;
		}
		++len;
	}

	return training_data.size();
}

// NOTE
//     : 基本的な思考としては、ある局面において指し手A, B, Cがあり、
//       それぞれ2, 5, 3回訪問されているとき、A, B, Cのprobabilityは、0.2, 0.5, 0.3 である。って感じかな？
// @template add
//     : false なら、初めての局面の時。
//       true なら重複局面(2回目以降の登場の時)なので、加算する(全部の局面読み込んだ後で平均をとる。)。
template <bool add>
inline void __visits_to_proberbility(
	TrainingData& _data, const std::vector<MoveVisits>& _candidates, const double temperature
) {
	if (_candidates.size() == 1) {
		// 候補手数が1 なら、確定でその指し手の確率 =1 (or += 1)
		// one-hot
		const auto& moveVisits = _candidates[0];
		if constexpr (add)
			_data.candidates[moveVisits.move16] += 1.0f;
		else
			_data.candidates[moveVisits.move16] = 1.0f;
	}
	else if (temperature == 0) {
		// temperature == 0が指定された時も、one-hot に(貪欲に)。
		// greedy
		const auto itr = std::max_element(
			_candidates.begin(), _candidates.end(), [](const MoveVisits& a, const MoveVisits& b) { return a.visitNum < b.visitNum; }
		);
		const MoveVisits& moveVisits = *itr;
		if constexpr (add)
			_data.candidates[moveVisits.move16] += 1.0f;
		else
			_data.candidates[moveVisits.move16] = 1.0f;
	}
	else if (temperature == 1) {
		// 普通に、確率 = 訪問回数 / 合計訪問回数とする。
		const float sum_visitNum = (float)std::accumulate(
			_candidates.begin(),
			_candidates.end(),
			0, [](int acc, const MoveVisits& move_visits) { return acc + move_visits.visitNum; }
		);

		if constexpr (!add) {
			assert(_data.candidates.size() == 0);
		}

		for (const auto& moveVisits : _candidates) {
			const float proberbility = (float)moveVisits.visitNum / sum_visitNum;
			if constexpr (add) {
				_data.candidates[moveVisits.move16] += proberbility;
			}
			else {
				_data.candidates[moveVisits.move16] = proberbility;
				// NOTE: 以下は不適。最後までloop 回し切ってからunlock しようっていう魂胆やと思うけど、これは二重lock獲得になるので不適。(lockして、unlockせずにlockする これは二重獲得で不適切。)
				//sync_cout << "[p]";
			}
		}
		//sync_cout << "e" << IO_UNLOCK;
	}
	else {
		double exponentiated_visits[593];
		double sum = 0;
		for (size_t i = 0; i < _candidates.size(); i++) {
			const auto& moveVisits = _candidates[i];
			const auto new_visits = std::pow(moveVisits.visitNum, 1.0 / temperature);

			exponentiated_visits[i] = new_visits;
			sum += new_visits;
		}
		for (size_t i = 0; i < _candidates.size(); i++) {
			const auto& moveVisits = _candidates[i];
			const float proberbility = (float)(exponentiated_visits[i] / sum);
			if constexpr (add)
				_data.candidates[moveVisits.move16] += proberbility;
			else
				_data.candidates[moveVisits.move16] = proberbility;
		}
	}
}

// フォーマット自動判別
inline bool __is_hcpe(std::ifstream& ifs) {
	if (ifs.tellg() % sizeof(HuffmanCodedPosAndEval) == 0) {
		// NOTE
		//     : https://learn.microsoft.com/ja-jp/cpp/error-messages/compiler-warnings/compiler-warning-level-2-c4146?view=msvc-170
		//       https://ja.stackoverflow.com/questions/83178/%E5%8D%98%E9%A0%85%E6%BC%94%E7%AE%97%E5%AD%90-%E3%82%92%E7%AC%A6%E5%8F%B7%E7%84%A1%E3%81%97%E6%95%B4%E6%95%B0%E5%9E%8B%E3%81%AB%E4%BD%BF%E7%94%A8%E3%81%97%E3%81%9F%E5%A0%B4%E5%90%88%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6
		//       符号なし整数に単項マイナス演算子を適用した結果は処理系定義らしい？
		//       まぁ兎に角やるべきでない。
		//     : sizeof() の戻り値は符号なし整数。
		// 最後のデータがhcpeであるかで判別
		ifs.seekg(
			-(long long int)sizeof(HuffmanCodedPosAndEval),
			std::ios_base::end
		);
		HuffmanCodedPosAndEval hcpe;
		ifs.read((char*)&hcpe, sizeof(HuffmanCodedPosAndEval));
		if (hcpe.hcp.isOK() && hcpe.bestMove16 >= 1 && hcpe.bestMove16 <= 26703) {
			return true;
		}
	}
	return false;
}

// ================================================== end

std::vector<TrainingData> trainingData;
// 重複チェック用 局面に対応するtrainingDataのインデックスを保持
std::unordered_map<HuffmanCodedPos, unsigned int> duplicates;

// make result
inline float make_result(const uint8_t result, const Color color) {
	const GameResult gameResult = (GameResult)(result & 0x3);
	if (gameResult == Draw)
		return 0.5f;

	if ((color == Black && gameResult == BlackWin) ||
		(color == White && gameResult == WhiteWin)) {
		return 1.0f;
	}
	else {
		return 0.0f;
	}
}

// hcpe形式の指し手をone-hotの方策として読み込む
// 複数回呼ぶことで、複数ファイルの読み込みが可能
// @arg use_average: 重複局面について、平均を取る。
inline size_t load_hcpe(
	std::ifstream& ifs, bool use_average, const double eval_scale, size_t& len
) {
	for (int p = 0; ifs; ++p) {
		HuffmanCodedPosAndEval hcpe;
		ifs.read((char*)&hcpe, sizeof(HuffmanCodedPosAndEval));
		if (ifs.eof()) {
			break;
		}

		const int eval = (int)(hcpe.eval * eval_scale);
		if (use_average) {
			auto ret = duplicates.emplace(hcpe.hcp, trainingData.size());
			if (ret.second) {
				auto& data = trainingData.emplace_back(
					hcpe.hcp,
					eval,
					make_result(hcpe.gameResult, hcpe.hcp.color())
				);
				data.candidates[hcpe.bestMove16] = 1;
			}
			else {
				// NOTE
				//     : https://cpprefjp.github.io/reference/unordered_map/unordered_map/emplace.html
				//       重複局面で合った場合、過去に登録した時の箇所へのiterator : ret.first
				//       であり、ret.first->second にはkey/value のvalue が入っており、
				//       そのvalue はduplicates の定義より、trainingData のインデックス。
				//       重複データの場合、加算する(hcpe3_decode_with_valueで平均にする)
				auto& data = trainingData[ret.first->second];
				data.eval += eval;
				data.result += make_result(hcpe.gameResult, hcpe.hcp.color());
				data.candidates[hcpe.bestMove16] += 1;    // NOTE: この局面において、hcpe.bestMove16 が指された回数 +1
				// hcpe3 化する時に役立つよね。
				data.count++;
			}
		}
		else {
			auto& data = trainingData.emplace_back(
				hcpe.hcp,
				eval,
				make_result(hcpe.gameResult, hcpe.hcp.color())
			);
			data.candidates[hcpe.bestMove16] = 1;
		}
		++len;
	}

	return trainingData.size();
}

// NOTE
//     : 基本的な思考としては、ある局面において指し手A, B, Cがあり、
//       それぞれ2, 5, 3回訪問されているとき、A, B, Cのprobabilityは、0.2, 0.5, 0.3 である。って感じかな？
// @template add
//     : false なら、初めての局面の時。
//       true なら重複局面(2回目以降の登場の時)なので、加算する(全部の局面読み込んだ後で平均をとる。)。
template <bool add>
inline void visits_to_proberbility(
	TrainingData& data, const std::vector<MoveVisits>& candidates, const double temperature
) {
	if (candidates.size() == 1) {
		// 候補手数が1 なら、確定でその指し手の確率 =1 (or += 1)
		// one-hot
		const auto& moveVisits = candidates[0];
		if constexpr (add)
			data.candidates[moveVisits.move16] += 1.0f;
		else
			data.candidates[moveVisits.move16] = 1.0f;
	}
	else if (temperature == 0) {
		// temperature == 0が指定された時も、one-hot に(貪欲に)。
		// greedy
		const auto itr = std::max_element(
			candidates.begin(), candidates.end(), [](const MoveVisits& a, const MoveVisits& b) { return a.visitNum < b.visitNum; }
		);
		const MoveVisits& moveVisits = *itr;
		if constexpr (add)
			data.candidates[moveVisits.move16] += 1.0f;
		else
			data.candidates[moveVisits.move16] = 1.0f;
	}
	else if (temperature == 1) {
		// 普通に、確率 = 訪問回数 / 合計訪問回数とする。
		const float sum_visitNum = (float)std::accumulate(
			candidates.begin(),
			candidates.end(),
			0, [](int acc, const MoveVisits& move_visits) { return acc + move_visits.visitNum; }
		);
		for (const auto& moveVisits : candidates) {
			const float proberbility = (float)moveVisits.visitNum / sum_visitNum;
			if constexpr (add)
				data.candidates[moveVisits.move16] += proberbility;
			else
				data.candidates[moveVisits.move16] = proberbility;
		}
	}
	else {
		double exponentiated_visits[593];
		double sum = 0;
		for (size_t i = 0; i < candidates.size(); i++) {
			const auto& moveVisits = candidates[i];
			const auto new_visits = std::pow(moveVisits.visitNum, 1.0 / temperature);

			exponentiated_visits[i] = new_visits;
			sum += new_visits;
		}
		for (size_t i = 0; i < candidates.size(); i++) {
			const auto& moveVisits = candidates[i];
			const float proberbility = (float)(exponentiated_visits[i] / sum);
			if constexpr (add)
				data.candidates[moveVisits.move16] += proberbility;
			else
				data.candidates[moveVisits.move16] = proberbility;
		}
	}
}

// フォーマット自動判別
inline bool is_hcpe(std::ifstream& ifs) {
	if (ifs.tellg() % sizeof(HuffmanCodedPosAndEval) == 0) {
		// NOTE
		//     : https://learn.microsoft.com/ja-jp/cpp/error-messages/compiler-warnings/compiler-warning-level-2-c4146?view=msvc-170
		//       https://ja.stackoverflow.com/questions/83178/%E5%8D%98%E9%A0%85%E6%BC%94%E7%AE%97%E5%AD%90-%E3%82%92%E7%AC%A6%E5%8F%B7%E7%84%A1%E3%81%97%E6%95%B4%E6%95%B0%E5%9E%8B%E3%81%AB%E4%BD%BF%E7%94%A8%E3%81%97%E3%81%9F%E5%A0%B4%E5%90%88%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6
		//       符号なし整数に単項マイナス演算子を適用した結果は処理系定義らしい？
		//       まぁ兎に角やるべきでない。
		//     : sizeof() の戻り値は符号なし整数。
		// 最後のデータがhcpeであるかで判別
		ifs.seekg(
			-(long long int)sizeof(HuffmanCodedPosAndEval),
			std::ios_base::end
		);
		HuffmanCodedPosAndEval hcpe;
		ifs.read((char*)&hcpe, sizeof(HuffmanCodedPosAndEval));
		if (hcpe.hcp.isOK() && hcpe.bestMove16 >= 1 && hcpe.bestMove16 <= 26703) {
			return true;
		}
	}
	return false;
}

// hcpe3形式のデータを読み込み、ランダムアクセス可能なように加工し、trainingDataに保存する
// 複数回呼ぶことで、複数ファイルの読み込みが可能
// @arg filepath: このファイルから教師を読み込む。
// @arg a
//     : a != 0 の時、評価値を756.0864962951762 / a倍する。
//       a == 0 の時、評価値は弄らない。
// @arg [out] len
//     : 実際の局面数。
//       恐らく、use_average を使うと重複局面は一つとしてカウントされるので、
//       実際に学習する局面数はtrainingData.size()で、ファイルに含まれている局面数を重複除去せずに数えた個数はこのlen。
// @return; trainingData のサイズ。(教師数)
//       
size_t __load_hcpe3(
	const std::string& filepath, bool use_average, double a, double temperature, size_t& len
) {
	std::ifstream ifs(filepath, std::ifstream::binary | std::ios::ate);
	if (!ifs) {
		std::cout << "Error: failed to open " << filepath << std::endl;
		return trainingData.size();
	}

	trainingData.reserve(40960000);
	duplicates.reserve(40960000);


	// eval_scale
	//     : 評価値に直接掛ける値。
	const double eval_scale = a == 0 ? 1 : 756.0864962951762 / a;

	// フォーマット自動判別
	// hcpeの場合は、指し手をone-hotの方策として読み込む
	if (is_hcpe(ifs)) {
		ifs.seekg(std::ios_base::beg);
		return load_hcpe(ifs, use_average, eval_scale, len);
	}
	ifs.seekg(std::ios_base::beg);

	std::vector<MoveVisits> candidates;    // NOTE: 毎局面ごとに、指し手と訪問回数が入る

#ifdef DEBUG
	// 1棋譜だけ読み込む。
	for (int p = 0; ifs && p < 1; ++p) {
#else
	for (int p = 0; ifs; ++p) {
#endif
		HuffmanCodedPosAndEval3 hcpe3;
		ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
		if (ifs.eof()) {
			break;
		}
		assert(hcpe3.moveNum <= 513);

		// 開始局面
		Position pos;
		if (!pos.set(hcpe3.hcp)) {
			std::stringstream ss("INCORRECT_HUFFMAN_CODE at ");
			ss << filepath << "(" << p << ")";
			throw std::runtime_error(ss.str());
		}
		StateListPtr states{ new std::deque<StateInfo>(1) };

		for (int i = 0; i < hcpe3.moveNum; ++i) {
			// NOTE: i番目(i + 1手目)の指し手を読み込む
			MoveInfo moveInfo;
			ifs.read((char*)&moveInfo, sizeof(MoveInfo));
			assert(moveInfo.candidateNum <= 593);    // NOTE: 一局面における候補手の数の最大値は593

			// candidateNum==0の手は読み飛ばす
			if (moveInfo.candidateNum > 0) {
				candidates.resize(moveInfo.candidateNum);
				ifs.read((char*)candidates.data(), sizeof(MoveVisits) * moveInfo.candidateNum);

				const auto hcp = pos.toHuffmanCodedPos();
				const int eval = (int)(moveInfo.eval * eval_scale);
				if (use_average) {
					// https://cpprefjp.github.io/reference/unordered_map/unordered_map/emplace.html
					auto ret = duplicates.emplace(hcp, trainingData.size());
					if (ret.second) {    // 今までに無かった、新しい局面
						// https://cpprefjp.github.io/reference/vector/vector/emplace_back.html
						// C++14 まで：なし
						// C++17 から：構築した要素への参照
						auto& data = trainingData.emplace_back(
							hcp,
							eval,
							make_result(hcpe3.result, pos.turn())
						);
						visits_to_proberbility<false>(data, candidates, temperature);
					}
					else {    // 重複局面
						// 重複データの場合、加算する(hcpe3_decode_with_valueで平均にする)
						auto& data = trainingData[ret.first->second];
						data.eval += eval;
						data.result += make_result(hcpe3.result, pos.turn());
						visits_to_proberbility<true>(data, candidates, temperature);
						data.count++;

					}
				}
				else {
					auto& data = trainingData.emplace_back(
						hcp,
						eval,
						make_result(hcpe3.result, pos.turn())
					);
					visits_to_proberbility<false>(data, candidates, temperature);
				}
				++len;
			}

			const Move move = move16toMove((Move)moveInfo.selectedMove16, pos);
			pos.doMove(move, states->emplace_back(StateInfo()));
		}
	}

	return trainingData.size();
}


size_t __only_read_hcpe3(
	std::vector<TrainingData>& training_data, std::unordered_map<HuffmanCodedPos, unsigned int>& duplicates,
	const std::string& filepath, const bool use_average, const double a, const double temperature, size_t& len
) {
	std::ifstream ifs(filepath, std::ifstream::binary | std::ios::ate);
	if (!ifs) {
		std::cout << "Error: failed to open " << filepath << std::endl;
		return training_data.size();
	}

	// eval_scale
	//     : 評価値に直接掛ける値。
	const double eval_scale = a == 0 ? 1 : 756.0864962951762 / a;

	// フォーマット自動判別
	// hcpeの場合は、指し手をone-hotの方策として読み込む
	if (__is_hcpe(ifs)) {
		ifs.seekg(std::ios_base::beg);
		return __load_hcpe(training_data, duplicates, ifs, use_average, eval_scale, len);
	}
	ifs.seekg(std::ios_base::beg);

	// 各プレイアウトの開始位置を格納
	std::vector<unsigned int> playout_pos;
	//std::streampos crrt_pos;

	std::vector<MoveVisits> candidates;    // NOTE: 毎局面ごとに、指し手と訪問回数が入る
	for (int p = 0; ifs; ++p) {
		playout_pos.emplace_back(ifs.tellg());
		HuffmanCodedPosAndEval3 hcpe3;
		ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
		// https://pc98.skr.jp/posts/2016/0419/
		if (ifs.eof()) {
			playout_pos.pop_back();    // まだあると思ったけど、無かったので削除。「さっき追加した奴は無しね！」
			break;
		}

		// NOTE: ++i は、一手進める事を意味する。
		for (int i = 0; i < hcpe3.moveNum; ++i) {
			MoveInfo moveInfo;
			ifs.read((char*)&moveInfo, sizeof(MoveInfo));
			assert(moveInfo.candidateNum <= 593);    // NOTE: 一局面における候補手の数の最大値は593
			if (moveInfo.candidateNum > 0) {
				// 候補手の数だけ、MoveVisits をbuffer に読み込む。
				candidates.resize(moveInfo.candidateNum);
				ifs.read((char*)candidates.data(), sizeof(MoveVisits) * moveInfo.candidateNum);
				++len;
			}
		}
	}
}

size_t __load_hcpe3_parallel(
	std::vector<TrainingData>&training_data, std::unordered_map<HuffmanCodedPos, unsigned int>&duplicates, const int n_threads,
	const std::string & filepath, const bool use_average, const double a, const double temperature, size_t & len
) {
	std::ifstream ifs(filepath, std::ifstream::binary | std::ios::ate);
	if (!ifs) {
		std::cout << "Error: failed to open " << filepath << std::endl;
		return training_data.size();
	}

	// eval_scale
	//     : 評価値に直接掛ける値。
	const double eval_scale = a == 0 ? 1 : 756.0864962951762 / a;

	// フォーマット自動判別
	// hcpeの場合は、指し手をone-hotの方策として読み込む
	if (__is_hcpe(ifs)) {
		ifs.seekg(std::ios_base::beg);
		return __load_hcpe(training_data, duplicates, ifs, use_average, eval_scale, len);
	}
	ifs.seekg(std::ios_base::beg);

	// 各プレイアウトの開始位置を格納
	std::vector<unsigned int> playout_pos;
	//std::streampos crrt_pos;

	//std::vector<MoveVisits> candidates;    // NOTE: 毎局面ごとに、指し手と訪問回数が入る
	//for (int p = 0; ifs; ++p) {
	//	playout_pos.emplace_back(ifs.tellg());
	//	HuffmanCodedPosAndEval3 hcpe3;
	//	ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
	//	// https://pc98.skr.jp/posts/2016/0419/
	//	if (ifs.eof()) {
	//		playout_pos.pop_back();    // まだあると思ったけど、無かったので削除。「さっき追加した奴は無しね！」
	//		break;
	//	}

	//	// NOTE: ++i は、一手進める事を意味する。
	//	for (int i = 0; i < hcpe3.moveNum; ++i) {
	//		MoveInfo moveInfo;
	//		ifs.read((char*)&moveInfo, sizeof(MoveInfo));
	//		auto crrt_pos = ifs.tellg();
	//		assert(moveInfo.candidateNum <= 593);    // NOTE: 一局面における候補手の数の最大値は593
	//		if (moveInfo.candidateNum > 0) {
	//			ifs.seekg(crrt_pos + static_cast<std::streampos>(sizeof(MoveVisits) * moveInfo.candidateNum), std::ios_base::beg);
	//			++len;
	//		}
	//	}
	//}

	// TODO: vector, unordered_map のスレッドセーフな挙動を調べる
	std::mutex mtx;
	std::vector<reading_hcpe3_task_t> task_queue;
	std::atomic<bool> stop = false;

	auto file_reader = [&mtx, &ifs, &len, &task_queue, &stop]() {
		std::vector<std::vector<MoveVisits>> _candidates_list;    // 一局分の情報が入る。
		std::vector<MoveInfo> _move_info_list;                    // 一局分の情報が入る

		for (int p = 0; ifs; ++p) {
			HuffmanCodedPosAndEval3 _hcpe3;
			ifs.read((char*)&_hcpe3, sizeof(HuffmanCodedPosAndEval3));
			if (ifs.eof()) {
				break;
			}
			if (!(_hcpe3.moveNum <= 513)) {
				std::cout << "Error: !(hcpe3.moveNum <= 513)" << std::endl;
			}

			_candidates_list.clear();
			_move_info_list.clear();

			for (int i = 0; i < _hcpe3.moveNum; ++i) {
				MoveInfo& _moveInfo = _move_info_list.emplace_back();
				ifs.read((char*)&_moveInfo, sizeof(MoveInfo));
				assert(_moveInfo.candidateNum <= 593);    // NOTE: 一局面における候補手の数の最大値は593

				// candidateNum==0の手は読み飛ばす
				if (_moveInfo.candidateNum > 0) {
					std::vector<MoveVisits>& _candidates = _candidates_list.emplace_back();
					_candidates.resize(_moveInfo.candidateNum);
					ifs.read((char*)_candidates.data(), sizeof(MoveVisits) * _moveInfo.candidateNum);

					++len;
				}
			}

			mtx.lock();
			task_queue.emplace_back(_hcpe3, _candidates_list, _move_info_list);
			mtx.unlock();
		}

		stop = true;
	};

	// HACK: len は、最終的には、atomic_len でカウントしておいてから、最後にlen = atomic_len; ってしよう。
	auto worker = [&n_threads, &eval_scale, &use_average, &temperature, &mtx, &task_queue, &stop, &duplicates, &training_data]() {
		while (1) {
			// https://yohhoy.hatenablog.jp/entry/2013/12/15/204116
			// 恐らく問題ない....?
			mtx.lock();
			if (!task_queue.empty()) {
				const auto _task = std::move(task_queue.back());
				task_queue.pop_back();
				mtx.unlock();

				const HuffmanCodedPosAndEval3& hcpe3 = _task.hcpe3;

				// 開始局面
				Position pos;
				if (!pos.set(hcpe3.hcp)) {
					std::stringstream ss("INCORRECT_HUFFMAN_CODE at ");
					std::cout << "Error: !pos.set(hcpe3.hcp)" << std::endl;
					throw std::runtime_error(ss.str());
				}
				StateListPtr states{ new std::deque<StateInfo>(1) };

				// candidateNum==0 の場合は読み飛ばされるので、単純にi でアクセスしてはならない
				int candidates_list_idx = 0;

				for (int i = 0; i < hcpe3.moveNum; ++i) {
					// NOTE: i番目(i + 1手目)の指し手を読み込む
					const MoveInfo& _moveInfo = _task.move_info_list[i];

					//auto& pos_mtx = GetPositionMutex(&pos);

					// candidateNum==0の手は読み飛ばす
					if (_moveInfo.candidateNum > 0) {
						// 候補手の数だけ、MoveVisits をbuffer に読み込む。
						const std::vector<MoveVisits>& _candidates = _task.candidates_list[candidates_list_idx];
						++candidates_list_idx;

						const auto hcp = pos.toHuffmanCodedPos();
						const int eval = (int)(_moveInfo.eval * eval_scale);
						if (use_average) {
							// https://cpprefjp.github.io/reference/unordered_map/unordered_map/emplace.html
							mtx.lock();
							auto ret = duplicates.emplace(hcp, training_data.size());
							if (ret.second) {    // 今までに無かった、新しい局面
								// https://cpprefjp.github.io/reference/vector/vector/emplace_back.html
								// C++14 まで：なし
								// C++17 から：構築した要素への参照
								auto& data = training_data.emplace_back(
									hcp,
									eval,
									__make_result(hcpe3.result, pos.turn())
								);
								mtx.unlock();

								__visits_to_proberbility<false>(data, _candidates, temperature);
							}
							else {    // 重複局面
								// 重複データの場合、加算する(hcpe3_decode_with_valueで平均にする)
								auto& data = training_data[ret.first->second];
								mtx.unlock();

								data.eval += eval;
								data.result += __make_result(hcpe3.result, pos.turn());
								__visits_to_proberbility<true>(data, _candidates, temperature);
								data.count++;

							}
						}
						else {
							mtx.lock();
							auto& data = training_data.emplace_back(
								hcp,
								eval,
								__make_result(hcpe3.result, pos.turn())
							);
							mtx.unlock();

							__visits_to_proberbility<false>(data, _candidates, temperature);
						}
					}

					const Move move = move16toMove((Move)_moveInfo.selectedMove16, pos);
					pos.doMove(move, states->emplace_back(StateInfo()));
				}
			}
			else {
				mtx.unlock();

				if (stop) {
					// 空で、且つ読み込みを終わっているなら終了
					break;
				}

				std::this_thread::sleep_for(std::chrono::milliseconds(n_threads));
			}
		}
	};

	std::vector<std::thread> ths(n_threads);
	std::thread main_th(file_reader);
	for (auto&& th : ths) {
		th = std::thread(worker);
	}

	main_th.join();
	for (auto&& th : ths) {
		th.join();
	}

	ifs.close();

	return training_data.size();
}


size_t __load_hcpe3_parallel2(
	std::vector<TrainingData>& training_data, std::unordered_map<HuffmanCodedPos, unsigned int>& duplicates, const int n_threads,
	const std::string& filepath, const bool use_average, const double a, const double temperature, size_t& len
) {
	std::ifstream ifs(filepath, std::ifstream::binary | std::ios::ate);
	if (!ifs) {
		std::cout << "Error: failed to open " << filepath << std::endl;
		return training_data.size();
	}

	// eval_scale
	//     : 評価値に直接掛ける値。
	const double eval_scale = a == 0 ? 1 : 756.0864962951762 / a;

	// フォーマット自動判別
	// hcpeの場合は、指し手をone-hotの方策として読み込む
	if (__is_hcpe(ifs)) {
		ifs.seekg(std::ios_base::beg);
		return __load_hcpe(training_data, duplicates, ifs, use_average, eval_scale, len);
	}
	ifs.seekg(std::ios_base::beg);

	// 各プレイアウトの開始位置を格納
	std::vector<unsigned int> playout_pos;
	//std::streampos crrt_pos;

	//std::vector<MoveVisits> candidates;    // NOTE: 毎局面ごとに、指し手と訪問回数が入る
	//for (int p = 0; ifs; ++p) {
	//	playout_pos.emplace_back(ifs.tellg());
	//	HuffmanCodedPosAndEval3 hcpe3;
	//	ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
	//	// https://pc98.skr.jp/posts/2016/0419/
	//	if (ifs.eof()) {
	//		playout_pos.pop_back();    // まだあると思ったけど、無かったので削除。「さっき追加した奴は無しね！」
	//		break;
	//	}

	//	// NOTE: ++i は、一手進める事を意味する。
	//	for (int i = 0; i < hcpe3.moveNum; ++i) {
	//		MoveInfo moveInfo;
	//		ifs.read((char*)&moveInfo, sizeof(MoveInfo));
	//		auto crrt_pos = ifs.tellg();
	//		assert(moveInfo.candidateNum <= 593);    // NOTE: 一局面における候補手の数の最大値は593
	//		if (moveInfo.candidateNum > 0) {
	//			ifs.seekg(crrt_pos + static_cast<std::streampos>(sizeof(MoveVisits) * moveInfo.candidateNum), std::ios_base::beg);
	//			++len;
	//		}
	//	}
	//}

	// TODO: vector, unordered_map のスレッドセーフな挙動を調べる
	std::mutex mtx;
	std::vector<reading_hcpe3_task_t> task_queue;
	std::atomic<int> queue_top = -1;
	//std::atomic<bool> stop = false;

	training_data.reserve(40960000);
	duplicates.reserve(40960000);

	auto print_queue_every_n = [&mtx, &queue_top, &training_data](const int n) {
		int n_last_queue_top = queue_top;
		while (1) {
			if (queue_top < 0) {
				break;
			}
			const auto& tmp = queue_top.load(std::memory_order_acquire);
			if (tmp < n_last_queue_top - n) {
				sync_cout << "[queue_top = " << tmp << "]" << IO_UNLOCK;
				//std::flush(std::cout);
				n_last_queue_top = tmp;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		sync_cout << sync_endl;
	};

	auto file_reader = [&mtx, &ifs, &len, &task_queue, &queue_top]() {

		for (int p = 0; ifs; ++p) {
			std::vector<std::vector<MoveVisits>> _candidates_list;    // 一局分の情報が入る。
			std::vector<MoveInfo> _move_info_list;                    // 一局分の情報が入る
			HuffmanCodedPosAndEval3 _hcpe3;

			ifs.read((char*)&_hcpe3, sizeof(HuffmanCodedPosAndEval3));
			if (ifs.eof()) {
				break;
			}
			if (!(_hcpe3.moveNum <= 513)) {
				std::cout << "Error: !(hcpe3.moveNum <= 513)" << std::endl;
			}

			_candidates_list.clear();
			_move_info_list.clear();

			for (int i = 0; i < _hcpe3.moveNum; ++i) {
				MoveInfo& _moveInfo = _move_info_list.emplace_back();
				ifs.read((char*)&_moveInfo, sizeof(MoveInfo));
				assert(_moveInfo.candidateNum <= 593);    // NOTE: 一局面における候補手の数の最大値は593

				// candidateNum==0の手は読み飛ばす
				if (_moveInfo.candidateNum > 0) {
					std::vector<MoveVisits>& _candidates = _candidates_list.emplace_back();
					_candidates.resize(_moveInfo.candidateNum);
					ifs.read((char*)_candidates.data(), sizeof(MoveVisits) * _moveInfo.candidateNum);

					++len;
				}
			}

			if (!(_hcpe3.moveNum == _move_info_list.size())) {
				std::cout << "Error: !(_hcpe3.moveNum == _move_info_list.size())" << std::endl;
				exit(1);
			}

			if (!(_move_info_list.size() >= _candidates_list.size())) {
				std::cout << "Error: !(_move_info_list.size() >= _candidates_list.size())" << std::endl;
				exit(1);
			}

			assert(_hcpe3.moveNum == _move_info_list.size());
			assert(_move_info_list.size() >= _candidates_list.size());

			//mtx.lock();
			task_queue.emplace_back(_hcpe3, _candidates_list, _move_info_list);
			//mtx.unlock();
		}

		//stop = true;
		queue_top = task_queue.size() - 1;
		};

    // NOTE: 恐らく。場所とかは違うくて良いから、とにかく、同じmutex を使ってれば、そいつをlock すれば、unlock されるまでは、そいつを他のスレッドはlock 出来ん。

	// HACK: len は、最終的には、atomic_len でカウントしておいてから、最後にlen = atomic_len; ってしよう。
	auto worker = [&n_threads, &eval_scale, &use_average, &temperature, &mtx, &task_queue, &queue_top, &duplicates, &training_data](
		const int thread_id
		) {
		// <point>
		//     : 他のスレッドと共有している変数は、見えない誰か(他のスレッド)に勝手に書き換えられる可能性があるので、
		//       共有する変数から値を観測した瞬間から一瞬でも時が経てば、共有する変数の値が何かは一意に定まらない(読み取った値と一致する保証が無い)。
		// NOTE
	    //     : ここでqueue_top >= 0 のチェックを行ってはならない。
		//       実際にtask_queue[crrt_queue_top] とアクセスする頃には、他のスレッドにデクリメントされて0 未満にされている可能性があるので。
		//       -> マルチスレッドでは、常に何者かに割り込まれている可能性を考慮しなければならない。
		//       -> 今回の場合で言うと、共有する変数を使う時はconst にload() するなりした方が無難である。
        //          共有している限り、何者かに(別のスレッドに)書き換えられている可能性があるので。
		while (true) {
			// https://yohhoy.hatenablog.jp/entry/2013/12/15/204116
			// 恐らく問題ない....?
			    

				//std::cout << "[" << thread_id << "," << queue_top - 1000 * (queue_top / 1000) << "]";
			    // NOTE
				//     : なぜここもlock ないと落ちるんだ？
				//       -> queue_top = 10 のthread より先に、queue_top = 9 のthread が"move, pop_back()" のブロックに入る可能性があり、
				//          この場合もまだ未使用のqueue_top = 10 のtask を先にmove してしまう。
				//       -> pop_back() 要らんやろ、取っ払っちゃえ!
				//mtx.lock();
				const auto&& crrt_queue_top = queue_top--;
				reading_hcpe3_task_t _task;
				if (crrt_queue_top >= 0) {
					// NOTE
					//     : ここにlock() が無いと、以下のように小さい数字の方が先にpop_back() に入ってしまうとおかしくなっちゃう。
					//       p1 [queue_top = 10]                                                            [_task = std::move(task_queue[10])]
					//       p2 [queue_top = 9]  [_task = std::move(task_queue[9])] [task_queue.poo_back()]
					//       -> そもそもpop_back() する必要ないのでは？末尾がそのままmove によって移動代入されるので、
					_task = std::move(task_queue[crrt_queue_top]);
					//task_queue.pop_back();
					//mtx.unlock();
				}
				else {
					//mtx.unlock();
					break;
				}

				const HuffmanCodedPosAndEval3& hcpe3 = _task.hcpe3;

				// 開始局面
				Position pos;
				if (!pos.set(hcpe3.hcp)) {
					std::stringstream ss("INCORRECT_HUFFMAN_CODE at ");
					//std::stringstream pss;
					//pos.print(pss);
					ss
						<< "[" << hcpe3.hcp.to_str()
						<< "][thread_id=" << thread_id
						<< "][queue_top=" << queue_top
						//<< "][move_info_list.size()" << _task.move_info_list.size()
						//<< "][candidates_list.size()" << _task.candidates_list.size()
						<< "]";
					std::cout << "Error: !pos.set(hcpe3.hcp)" << std::endl;
					throw std::runtime_error(ss.str());
				}
				StateListPtr states{ new std::deque<StateInfo>(1) };


				// candidateNum==0 の場合は読み飛ばされるので、単純にi でアクセスしてはならない
				int candidates_list_idx = 0;
				
				// NOTE
				//     : 単純にpos_mtx でlock してもだめ。何故なら、vector は異なる要素へのアクセスはスレッドセーフだけど、
				//       emplace みたいな非const メンバ関数はスレッドセーフじゃないので、異なる局面であっても、同時にemplace()呼び出すだけでアウト。
				auto& pos_mtx = __GetPositionMutex(&pos);

				for (int i = 0; i < hcpe3.moveNum; ++i) {
					// NOTE: i番目(i + 1手目)の指し手を読み込む
					const MoveInfo& _moveInfo = _task.move_info_list[i];

					// candidateNum==0の手は読み飛ばす
					if (_moveInfo.candidateNum > 0) {
						// 候補手の数だけ、MoveVisits をbuffer に読み込む。
						const std::vector<MoveVisits>& _candidates = _task.candidates_list[candidates_list_idx];
						++candidates_list_idx;

						const auto hcp = pos.toHuffmanCodedPos();
						const int eval = (int)(_moveInfo.eval * eval_scale);

						if (use_average) {
							// https://cpprefjp.github.io/reference/unordered_map/unordered_map/emplace.html
							//std::cout << "l";
							//
							//if (i % 20 == 0) {
							//	sync_cout
							//		<< "[auto ret = duplicates.emplace(hcp, training_data.size())]"
							//		<< "[" << i << "," << hcpe3.moveNum - 1 << "]"
							//		<< "[" << thread_id << "] : queue_top = " << queue_top << sync_endl;
							//}

							mtx.lock();
							auto ret = duplicates.emplace(hcp, training_data.size());
							if (ret.second) {    // 今までに無かった、新しい局面
								// https://cpprefjp.github.io/reference/vector/vector/emplace_back.html
								// C++14 まで：なし
								// C++17 から：構築した要素への参照
								//std::cout << "e";
								// https://cpprefjp.github.io/reference/vector/vector/emplace_back.html
								auto& data = training_data.emplace_back(
									hcp,
									eval,
									__make_result(hcpe3.result, pos.turn())
								);
								mtx.unlock();
								//std::cout << "A" << i;

								// NOTE
								//     : data 、つまりtraininig_data の一要素を弄るだけなので、全く同じ要素に同時にアクセスしない限り問題ない。
								//       -> 同時にアクセスする可能性があるのは、pos のhashKey が一致する時。(同じ局面の可能性あり)
								pos_mtx.lock();
								// 今までに読み込んだファイル数が増えるにつれて、それらは定期的にclear されてるはずなのに、
								// ここが特にどんどん遅くなる。ただ、上のtraining_data.emplace_back()なんかも遅くなってる、
								__visits_to_proberbility<false>(data, _candidates, temperature);
								pos_mtx.unlock();
							}
							else {    // 重複局面
								mtx.unlock();
								pos_mtx.lock();
								// 重複データの場合、加算する(hcpe3_decode_with_valueで平均にする)
								auto& data = training_data[ret.first->second];
								//std::cout << "B" << i;

								data.eval += eval;
								data.result += __make_result(hcpe3.result, pos.turn());
								__visits_to_proberbility<true>(data, _candidates, temperature);
								data.count++;
								pos_mtx.unlock();
							}
						}
						else {
							pos_mtx.lock();
							auto& data = training_data.emplace_back(
								hcp,
								eval,
								__make_result(hcpe3.result, pos.turn())
							);

							__visits_to_proberbility<false>(data, _candidates, temperature);
							pos_mtx.unlock();
						}
					}

					const Move move = move16toMove((Move)_moveInfo.selectedMove16, pos);
					pos.doMove(move, states->emplace_back(StateInfo()));
				}

				//sync_cout << sync_endl;
		}
	};

	stopwatch_t sw;
	sw.start_with_print("read()");
	std::thread main_th(file_reader);
	main_th.join();
	sw.stop_with_print();
	
	std::cout << "[queue_top = " << queue_top << "]";

	std::thread back(print_queue_every_n, 40000);
	std::vector<std::thread> ths(n_threads);
	// https://stackoverflow.com/questions/3074646/how-to-catch-and-print-an-exception-of-unkown-type
	try {
		int i = 0;
		for (auto&& th : ths) {
			th = std::thread(worker, i);
			++i;
		}
	}
	catch (const std::exception& ex) {
		std::cout << "[ex=" << ex.what() << "]" << std::endl;
	}
	catch (const std::string& ex) {
		std::cout << "[ex=" << ex << "]" << std::endl;
	}

	for (auto&& th : ths) {
		th.join();
	}
	back.join();

	ifs.close();

	return training_data.size();
}

int load_hcpex_test0(const std::string & filename) {
	size_t len;
	//std::cout << "info: __load_hcpe3_parallel() start" << std::endl;

	//auto size = __load_hcpe3(filename, true, 0, 1, len);
	
	//stopwatch_t sw;
	//sw.start_with_print("__load_hcpe3()");
	//auto size = __load_hcpe3(filename, true, 0, 1, len);
	//sw.stop_with_print();

	// 3スレッドで最速
	// 

	stopwatch_t sw;
	sw.start_with_print("__load_hcpe3_parallel()");
	auto size = __load_hcpe3_parallel2(trainingData, duplicates, 3, filename, true, 0, 1, len);
	sw.stop_with_print();
	
	
	//std::cout << "info: __load_hcpe3_parallel() done" << std::endl;

	std::cout << "info: size = " << size << std::endl;
	std::cout << "info: trainingData.size() = " << trainingData.size() << std::endl;

	//// 読み込んだ内容を表示
	//__Board board;

	//int i = 1;
	//for (const auto& data : trainingData) {
	//	const auto& hcp = data.hcp;
	//	const auto& eval = data.eval;
	//	const auto& result = data.result;
	//	const auto& candidates = data.candidates;
	//	const auto& count = data.count;

	//	board.set_hcp(hcp);

	//	std::cout << "[" << i << "/" << trainingData.size() << "] : " << board.toSFEN() << std::endl;
	//	std::cout << "    " << "eval = " << eval << ", result = " << result << ", count = " << count << std::endl;
	//	for (const auto& kv : candidates) {
	//		std::cout << "    " << __move_to_usi(kv.first)
	//			<< "[" << __dlshogi_make_move_label(kv.first, board.turn()) << "] = " << kv.second << std::endl;
	//	}
	//	++i;
	//}

	return 1;
}