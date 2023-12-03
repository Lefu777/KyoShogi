#include "dataloader.h"

static constexpr int SFEN_BUF_SIZE = 1024;

constexpr uint64_t MUTEX_NUM = 65536; // must be 2^n
std::mutex mutexes[MUTEX_NUM];
inline std::mutex& GetPositionMutex(const Position* pos)
{
	// NOTE
	//     : pos->getKey() & (MUTEX_NUM - 1) が簡易的なHash になってる感じ？
	//       それによって、同じ局面を同時に操作しない、的な...?
	return mutexes[pos->getKey() & (MUTEX_NUM - 1)];
}


// NOTE
//     : https://skpme.com/635/
//       分割コンパイルする場合、staticメンバ変数の定義では、static を付けない。

void __FastDataloader::print_teachers() const {
	print_teachers(_teachers);
}

void __FastDataloader::print_teachers(const std::vector<teacher_t>& teachers) {
	int i = 0;
	for (const auto& e : teachers) {
		std::cout << "[" << i << "][" << e.sfen << "][" << e.ply << "]" << std::endl;
		++i;
	}
}

// @arg dir: このフォルダ(ディレクトリ)にあるファイルを全て読み込む
// @arg[out] file_name: こいつに、ファイル一覧を格納。
// @return: 成功すると1, 失敗すると0
int __FastDataloader::_get_file_list(std::vector<std::string>& file_names, std::string dir) {
	using namespace std::filesystem;

	struct stat statDirectory;
	if (stat(dir.c_str(), &statDirectory) != 0) {
		std::cout << "Error: dir dose not exist, dir = " << dir << std::endl << std::flush;
		return 0;
	}

	directory_iterator iter(dir), end;
	std::error_code err;

	for (; iter != end && !err; iter.increment(err)) {
		const directory_entry entry = *iter;

		file_names.push_back(entry.path().string());
		//std::cout << "info: [" << file_names.back() << "]" << std::endl;
	}

	/* エラー処理 */
	if (err) {
		std::cout << err.value() << std::endl;
		std::cout << err.message() << std::endl;
		return 0;
	}
	return 1;
}

// 1file の教師データを読み込む
// @arg file_path: 読み込むファイルへのpath
// @arg[out] teachers: 読み込んだ教師を格納する
// @return: 成功すると1, 失敗すると0
int __FastDataloader::_read_one_file(std::vector<teacher_t>& teachers, std::string file_path) {
	FILE* file;

	// ファイルから読み込む時に、読み込んだデータを格納する変数。
	size_t sfen_size;
	char sfen_char[SFEN_BUF_SIZE];
	int move;
	int ply;
	float value;
	float result;

	if ((fopen_s(&file, file_path.c_str(), "rb")) != 0) {
		// https://qiita.com/izuki_y/items/26bf20c4b3b3750ab7a7
		std::cout << "Error: failed to open." << std::endl;
		return 0;
	}
	// https://oshiete.goo.ne.jp/qa/7050751.html
	while (true) {
		auto rsize = fread(&sfen_size, sizeof(size_t), 1, file);
		if (rsize != 1) {    // 読み込みエラー or 終端
			break;
		}
		if (sfen_size > SFEN_BUF_SIZE) {
			std::cout << "Error: run out of buffer size (sfen_size = " << sfen_size << " > SFEN_BUF_SIZE)" << std::endl;
			return 0;
		}
		fread(sfen_char, sizeof(char), sfen_size, file);
		fread(&move, sizeof(int), 1, file);
		fread(&ply, sizeof(int), 1, file);
		fread(&value, sizeof(float), 1, file);
		fread(&result, sizeof(float), 1, file);
		std::string sfen_str(sfen_char, sfen_size);
		teachers.emplace_back(sfen_str, move, ply, value, result);
	}

	if (!feof(file)) {
		std::cout << "Error: failed to parse " << file_path << std::endl;
		return 0;
	}


	fclose(file);
	return 1;
}

// make result
inline float _make_result(const uint8_t result, const Color color) {
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
size_t _load_hcpe(
	std::vector<TrainingData>& training_data, 
	std::unordered_map<HuffmanCodedPos, unsigned int>& duplicates,
	std::ifstream& ifs, const bool use_average, const double eval_scale, size_t& len
) {

	training_data.reserve(hcpe_data_reserve_size);
	duplicates.reserve(hcpe_data_reserve_size);

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
					_make_result(hcpe.gameResult, hcpe.hcp.color()),
					hcpe_TrainingData_candidates_reserve_size
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
				data.result += _make_result(hcpe.gameResult, hcpe.hcp.color());
				data.candidates[hcpe.bestMove16] += 1;    // NOTE: この局面において、hcpe.bestMove16 が指された回数 +1
				// hcpe3 化する時に役立つよね。
				data.count++;
			}
		}
		else {
			auto& data = training_data.emplace_back(
				hcpe.hcp,
				eval,
				_make_result(hcpe.gameResult, hcpe.hcp.color()),
				hcpe_TrainingData_candidates_reserve_size
			);
			data.candidates[hcpe.bestMove16] = 1;
		}
		++len;
	}

	ifs.close();

	return training_data.size();
}

//static stopwatch_t sw_visits_to_proberbility0;
//static stopwatch_t sw_visits_to_proberbility1;

// NOTE
//     : 基本的な思考としては、ある局面において指し手A, B, Cがあり、
//       それぞれ2, 5, 3回訪問されているとき、A, B, Cのprobabilityは、0.2, 0.5, 0.3 である。って感じかな？
// @template add
//     : false なら、初めての局面の時。
//       true なら重複局面(2回目以降の登場の時)なので、加算する(全部の局面読み込んだ後で平均をとる。)。
template <bool add>
void _visits_to_proberbility(
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
#ifdef DEBUG_
		sw_visits_to_proberbility0.resume();
#endif
		const float sum_visitNum = (float)std::accumulate(
			candidates.begin(),
			candidates.end(),
			0, [](int acc, const MoveVisits& move_visits) { return acc + move_visits.visitNum; }
		);
#ifdef DEBUG_
		sw_visits_to_proberbility0.pause();
		sw_visits_to_proberbility1.resume();
#endif
		for (const auto& moveVisits : candidates) {
			const float proberbility = (float)moveVisits.visitNum / sum_visitNum;
			if constexpr (add)
				data.candidates[moveVisits.move16] += proberbility;
			else
				data.candidates[moveVisits.move16] = proberbility;
		}
#ifdef DEBUG_
		sw_visits_to_proberbility1.pause();
#endif
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
bool _is_hcpe(std::ifstream& ifs) {
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
size_t _load_hcpe3(
	std::vector<TrainingData>& training_data,
	std::unordered_map<HuffmanCodedPos, unsigned int>& duplicates,
	const std::string& filepath, bool use_average, double a, double temperature, size_t& len
) {
	std::ifstream ifs(filepath, std::ifstream::binary | std::ios::ate);
	if (!ifs) {
		std::cout << "Error: failed to open " << filepath << std::endl;
		return training_data.size();
	}

	//stopwatch_t sw_reserve;
	//stopwatch_t sw0;
	//stopwatch_t sw1;
	//stopwatch_t sw2;
	//stopwatch_t sw3;



	//std::cout << "[training_data.capacity() = " << training_data.capacity() << "]" << std::endl;

	// eval_scale
	//     : 評価値に直接掛ける値。
	const double eval_scale = a == 0 ? 1 : 756.0864962951762 / a;

	// フォーマット自動判別
	// hcpeの場合は、指し手をone-hotの方策として読み込む
	if (_is_hcpe(ifs)) {
		ifs.seekg(std::ios_base::beg);
		return _load_hcpe(training_data, duplicates, ifs, use_average, eval_scale, len);
	}
	ifs.seekg(std::ios_base::beg);

	//sw_reserve.start_with_print("    reseve()");
	training_data.reserve(hcpe3_data_reserve_size);
	duplicates.reserve(hcpe3_data_reserve_size);
	//sw_reserve.stop_with_print();

	std::vector<MoveVisits> candidates;    // NOTE: 毎局面ごとに、指し手と訪問回数が入る
	candidates.reserve(593);

	//std::cout << "[training_data.capacity() = " << training_data.capacity() << "]" << std::endl;


	//sw0.start_pause_with_print("    sw_0");
	//sw1.start_pause_with_print("    sw_1");
	//sw2.start_pause_with_print("    sw_2");
	//sw3.start_pause_with_print("    sw_3");

	//sw_visits_to_proberbility0.start_pause_with_print("    vtp_0");
	//sw_visits_to_proberbility1.start_pause_with_print("    vtp_1");

	for (int p = 0; ifs; ++p) {
		HuffmanCodedPosAndEval3 hcpe3;
		ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
		if (ifs.eof()) {
			break;
		}
		if (!(hcpe3.moveNum <= 513)) {
			std::cout << "Error: !(hcpe3.moveNum <= 513)" << std::endl;
		}
		//assert(hcpe3.moveNum <= 513);

		// 開始局面
		Position pos;
		if (!pos.set(hcpe3.hcp)) {
			std::stringstream ss("INCORRECT_HUFFMAN_CODE at ");
			ss << filepath << "(" << p << ")";
			std::cout << "Error: !pos.set(hcpe3.hcp)" << std::endl;
			throw std::runtime_error(ss.str());
		}
		StateListPtr states{ new std::deque<StateInfo>(1) };


	    // NOTE: ここまではあまり時間食ってない。まぁ当然。(500ms)

		//if (debug && p % 10 == 0) std::cout << "p";

		// NOTE: ++i は、一手進める事を意味する。
		for (int i = 0; i < hcpe3.moveNum; ++i) {
			// NOTE: i番目(i + 1手目)の指し手を読み込む
			//if (debug) std::cout << "M";


			// NOTE: このread は全部で800ms ぐらい
			MoveInfo moveInfo;
			ifs.read((char*)&moveInfo, sizeof(MoveInfo));
			assert(moveInfo.candidateNum <= 593);    // NOTE: 一局面における候補手の数の最大値は593
			

			// candidateNum==0の手は読み飛ばす
			if (moveInfo.candidateNum > 0) {
				//sw0.resume();

				// 候補手の数だけ、MoveVisits をbuffer に読み込む。
				//std::cout << "r";
				candidates.resize(moveInfo.candidateNum);
				ifs.read((char*)candidates.data(), sizeof(MoveVisits) * moveInfo.candidateNum);

				const auto hcp = pos.toHuffmanCodedPos();
				const int eval = (int)(moveInfo.eval * eval_scale);
				if (use_average) {
					// https://cpprefjp.github.io/reference/unordered_map/unordered_map/emplace.html
					//if (debug) std::cout << "e";
					auto ret = duplicates.emplace(hcp, training_data.size());
					//sw0.pause();
					if (ret.second) {    // 今までに無かった、新しい局面
						//sw1.resume();
						// https://cpprefjp.github.io/reference/vector/vector/emplace_back.html
						// C++14 まで：なし
						// C++17 から：構築した要素への参照
						//std::cout << "b";
						auto& data = training_data.emplace_back(
							hcp,
							eval,
							_make_result(hcpe3.result, pos.turn()),
							hcpe3_TrainingData_candidates_reserve_size
						);
						//sw1.pause();
						//sw2.resume();
						_visits_to_proberbility<false>(data, candidates, temperature);
						//sw2.pause();
					}
					else {    // 重複局面
						//sw3.resume();

						// 重複データの場合、加算する(hcpe3_decode_with_valueで平均にする)
						auto& data = training_data[ret.first->second];
						data.eval += eval;
						data.result += _make_result(hcpe3.result, pos.turn());
						_visits_to_proberbility<true>(data, candidates, temperature);
						data.count++;

						//sw3.pause();
					}
				}
				else {
					auto& data = training_data.emplace_back(
						hcp,
						eval,
						_make_result(hcpe3.result, pos.turn()),
						hcpe3_TrainingData_candidates_reserve_size
					);
					_visits_to_proberbility<false>(data, candidates, temperature);
				}
				++len;

			}

			// NOTE: 以下のdoMove は全体で2000ms程度
			const Move move = move16toMove((Move)moveInfo.selectedMove16, pos);
			pos.doMove(move, states->emplace_back(StateInfo()));
		}
		//if (debug) std::cout << std::endl;


	}

	//sw0.stop_with_print();
	//sw1.stop_with_print();
	//sw2.stop_with_print();
	//sw3.stop_with_print();
	//sw_visits_to_proberbility0.stop_with_print();
	//sw_visits_to_proberbility1.stop_with_print();

	ifs.close();

	return training_data.size();
}

size_t _load_hcpe3_parallel2(
	std::vector<TrainingData>& training_data,
	std::unordered_map<HuffmanCodedPos, unsigned int>& duplicates,
	const int n_threads,
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
	if (_is_hcpe(ifs)) {
		ifs.seekg(std::ios_base::beg);
		return _load_hcpe(training_data, duplicates, ifs, use_average, eval_scale, len);
	}
	ifs.seekg(std::ios_base::beg);

	training_data.reserve(hcpe3_data_reserve_size);
	duplicates.reserve(hcpe3_data_reserve_size);

	// 各プレイアウトの開始位置を格納
	//std::vector<unsigned int> playout_pos;
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


	auto print_queue_every_n = [&mtx, &queue_top, &training_data](const int n) {
		int n_last_queue_top = queue_top;
		while (1) {
			if (queue_top < 0) {
				break;
			}
			const auto& tmp = queue_top.load(std::memory_order_acquire);
			if (tmp < n_last_queue_top - n) {
				sync_cout << "[" << tmp << "]" << IO_UNLOCK;
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

			//if (!(_hcpe3.moveNum == _move_info_list.size())) {
			//	std::cout << "Error: !(_hcpe3.moveNum == _move_info_list.size())" << std::endl;
			//	exit(1);
			//}

			//if (!(_move_info_list.size() >= _candidates_list.size())) {
			//	std::cout << "Error: !(_move_info_list.size() >= _candidates_list.size())" << std::endl;
			//	exit(1);
			//}

			assert(_hcpe3.moveNum == _move_info_list.size());
			assert(_move_info_list.size() >= _candidates_list.size());

			//mtx.lock();
			task_queue.emplace_back(_hcpe3, _candidates_list, _move_info_list);
			//mtx.unlock();
		}

		//stop = true;
		queue_top = task_queue.size() - 1;
		};

	// HACK: len は、最終的には、atomic_len でカウントしておいてから、最後にlen = atomic_len; ってしよう。
	auto worker = [&n_threads, &eval_scale, &use_average, &temperature, &mtx, &task_queue, &queue_top, &duplicates, &training_data](
		const int thread_id
		) {
			while (true) {

				//mtx.lock();
				const auto&& crrt_queue_top = queue_top--;
				reading_hcpe3_task_t _task;
				if (crrt_queue_top >= 0) {
					_task = std::move(task_queue[crrt_queue_top]);
					//task_queue.pop_back();    // TODO: 削除しなくても良いのかね？(メモリ使用量見る感じ良さげ)
					//mtx.unlock();
				}
				else {
					sync_cout << "[" << thread_id << ":done], crrt_queue_top = " << crrt_queue_top << sync_endl;
					//mtx.unlock();
					break;
				}

				const HuffmanCodedPosAndEval3& hcpe3 = _task.hcpe3;

				// 開始局面
				Position pos;
				if (!pos.set(hcpe3.hcp)) {
					std::stringstream ss("INCORRECT_HUFFMAN_CODE at ");
					std::cout << "Error: !pos.set(hcpe3.hcp)" << std::endl;
					throw std::runtime_error(ss.str());
				}
				StateListPtr states{ new std::deque<StateInfo>(1) };

				// NOTE
				//     : 単純にpos_mtx でlock してもだめ。何故なら、vector は異なる要素へのアクセスはスレッドセーフだけど、
				//       emplace みたいな非const メンバ関数はスレッドセーフじゃないので、異なる局面であっても、同時にemplace()呼び出すだけでアウト。
				auto& pos_mtx = GetPositionMutex(&pos);

				// candidateNum==0 の場合は読み飛ばされるので、単純にi でアクセスしてはならない
				int candidates_list_idx = 0;

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
							mtx.lock();

							// https://cpprefjp.github.io/reference/unordered_map/unordered_map/emplace.html
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
									_make_result(hcpe3.result, pos.turn()),
									hcpe3_TrainingData_candidates_reserve_size
								);
								//const auto&& tmp = training_data.size();
								//if (tmp % 5000000 == 0 || tmp < 8) {
								//	sync_cout << "[" << thread_id << "], training_data.size() = " << tmp << sync_endl;
								//}
								//std::cout << "A" << i;
								mtx.unlock();

								pos_mtx.lock();
								_visits_to_proberbility<false>(data, _candidates, temperature);
								pos_mtx.unlock();
							}
							else {    // 重複局面
								mtx.unlock();
								pos_mtx.lock();
								// 重複データの場合、加算する(hcpe3_decode_with_valueで平均にする)
								auto& data = training_data[ret.first->second];
								//std::cout << "B" << i;

								data.eval += eval;
								data.result += _make_result(hcpe3.result, pos.turn());
								_visits_to_proberbility<true>(data, _candidates, temperature);
								data.count++;

								pos_mtx.unlock();
							}
						}
						else {
							mtx.lock();
							auto& data = training_data.emplace_back(
								hcp,
								eval,
								_make_result(hcpe3.result, pos.turn()),
								hcpe3_TrainingData_candidates_reserve_size
							);
							mtx.unlock();

							// NOTE: 同じ局面であっても毎度違うところに格納するので、このdata にアクセスるスレッドは今のこのスレッドただ一つ
							_visits_to_proberbility<false>(data, _candidates, temperature);
						}
					}

					const Move move = move16toMove((Move)_moveInfo.selectedMove16, pos);
					pos.doMove(move, states->emplace_back(StateInfo()));
				}

				//sync_cout << sync_endl;
			}
		};

	stopwatch_t sw;
	//sw.start_with_print("read()");
	//std::thread main_th(file_reader);
	//main_th.join();
	file_reader();
	//sw.stop_with_print();

	std::cout << "queue_top = [" << queue_top << "]";

	std::thread print_progress(print_queue_every_n, 40000);
	std::vector<std::thread> ths(n_threads);
	int i = 0;
	for (auto&& th : ths) {
		th = std::thread(worker, i);
		++i;
	}

	for (auto&& th : ths) {
		th.join();
	}
	print_progress.join();

	ifs.close();

	return training_data.size();
}


// @return: 成功したら1, 失敗したら0
int __FastDataloader::_read_one_file(
	std::vector<TrainingData>& training_data,
	std::unordered_map<HuffmanCodedPos, unsigned int>& duplicates,
	std::string file_path
) {
	size_t len = 0;

	//__only_read_hcpe3(training_data, duplicates, file_path, true, 0, 1, len);

	//stopwatch_t sw;
	//sw.start_with_print("_load_hcpe3()");
	//_load_hcpe3(training_data, duplicates, file_path, true, 0, 1, len);
	//sw.stop_with_print();

	stopwatch_t sw;
	sw.start_with_print("_load_hcpe3_parallel2()");
	_load_hcpe3_parallel2(training_data, duplicates, 3, file_path, true, 0, 1, len);
	sw.stop_with_print();

	if (len == 0) {
		std::cout
			<< "Error: len = 0, there is no valid data or this file does not exist, file_path = " << file_path << std::endl;
		return 0;
	}
	return 1;
}


// _dir にある教師ファイルを全て読み込んで_teachers に格納する。
// ディレクトリにある、hcpe, hcpe3, bin を読み込む。
// @return: 成功したら1, 失敗したら0
int __FastDataloader::_read_files_all(
	std::vector<teacher_t>& teachers,
	std::vector<TrainingData>& training_data,
	std::unordered_map<HuffmanCodedPos, unsigned int>& duplicates,
	std::string dir
) {
	std::vector<std::string> file_names;
	//std::cout << "cinfo: start get_file_list()" << std::endl << std::flush;
	if (!_get_file_list(file_names, dir)) {
		std::cout << "Error: failed to get file_list, dir = " << dir << std::endl << std::flush;
		return 0;
	}

	const auto& n_files = file_names.size();
	if (n_files <= 0) {
		std::cout << "Error: file_names.size() = " << file_names.size() << " <= 0" << std::endl << std::flush;
		return 0;
	}

	int i = 0;
	for (const auto& file_name : file_names) {
		//std::cout << "cinfo: start read_one_file()" << std::endl << std::flush;

		std::cout << "info: reading[" << i + 1 << "/" << n_files << "] " << file_name << std::endl;
		if (file_extension_is_ok<teacher_t>(file_name)) {
			if (!_read_one_file(teachers, file_name)) {
				std::cout << "Error: failed to read, file_name = " << file_name << std::endl;
			}
			//std::cout << "info: teachers.size() = " << teachers.size() << std::endl;
		}
		else if (file_extension_is_ok<TrainingData>(file_name)) {
			if (!_read_one_file(training_data, duplicates, file_name)) {
				std::cout << "Error: failed to read, file_name = " << file_name << std::endl;
			}
			std::cout << "info: training_data.size() = " << training_data.size() << std::endl;
		}
		else {
			std::cout << "Error: " << file_name << " has an inappropriate extension, so passed reading." << std::endl;
		}
		++i;
	}
	return 1;
}

// _dir にある教師ファイルを全て読み込んで_teachers に格納する。
// 一番初めにこれで教師を読み込んだら、後はひたすらstore_teachers() するだけ。
// TODO: .bin のみを読み込むように
// @return: 成功したら1, 失敗したら0
int __FastDataloader::read_files_all() {
	//std::cout << "cinfo: start read_files_all()" << std::endl << std::flush;
	if (!_read_files_all(_teachers, _training_data, _duplicates, _dir)) {
		std::cout << "Error: failed to read_files_all()" << std::endl << std::flush;
		return 0;
	}

	// 指定したディレクトリに、二つのフォーマットの教師が入っている場合はエラー
	if (_teachers.size() != 0 && _training_data.size() != 0) {
		std::cout << "Error: mixed format." << std::endl;
		std::cout << "Error: This program is only for groups of teachers with only form A or only form B." << std::endl;
		return 0;
	}

	return 1;
}

// @arg src_dir: shuffle する教師が入ってるディレクトリ
// @arg dst_dir: shuffle した教師を書き出すディレクトリ
// @arg base_file_name
//     : shuffle した教師ファイルを書き出す時のベースとなる名前。
//     : ex> base_file_name = "sfen"
//       の時、[dst_dir]/sfen_[random文字列].bin に書き出される。
int __FastDataloader::shuffle_files(std::string src_dir, std::string dst_dir, std::string base_file_name) {
	std::vector<std::string> file_names;
	if (!_get_file_list(file_names, src_dir)) {
		std::cout << "Error: failed to get file_list, src_dir = " << src_dir << std::endl;
	}

	int i = 0;
	for (const auto& file_name : file_names) {
		if (!file_extension_is_ok<teacher_t>(file_name)) {
			std::cout << "Error: " << file_name << "  has an inappropriate extension, so passed reading." << std::endl;
		}
		else {
			std::cout << "info: reading[" << i << "] " << file_name << std::endl;
			if (!shuffle_write_one_file(file_name, dst_dir, base_file_name)) {
				std::cout << "Error: failed to shuffle, file_name = " << file_name << std::endl;
			}
		}
		++i;
	}
	return 1;
}

// @arg file_path: 読み込む教師ファイル名。こいつをshuffle する。
// @arg dst_dir
//     : このディレクトリに、shuffle した教師ファイルを書き出す。
//     : ex> data/shuffle
// @arg base_file_name
//     : 保存するファイル名
//     : ex> sfen
//       この時、data/shuffle/sfen.bin に保存。
int __FastDataloader::shuffle_write_one_file(std::string file_path, std::string dst_dir, std::string base_file_name) {
	std::vector<teacher_t> teachers;
	if (!_read_one_file(teachers, file_path)) {
		std::cout << "Error: failed to read " << file_path << std::endl;
		return 0;
	}

	shuffle(teachers);

	write_teacher(
		teachers,
		dst_dir + "/" + base_file_name,
		0,    // 書き出し中の表示を更新する間隔(のはずだったが、現状使ってない。)
		true,
		true
	);
	return 1;
}

// TODO
//     : こいつを汎用性のある奴にしたい。コイツあったらall ok みたいな。
//     : ここで、store_teachers() の、start_idx を貰って、それ以降のデータのみの新しいvectorを作る感じ？
//     -> あれか、reuse_start_idx みたいな感じで貰うか。
//     -> でもあれよな、random_pick は出来んくなるな。まぁしゃーないか。
//     -> 本当はキャッシュが最強なんでしょうね。
//     : _file_names_with_flag, _next_idx を初期化する関数！！！
//       __iter__() とかで良いんじゃね、名前。
// データが不足して初めて呼び出す。(※一番初めは当然呼び出す。)
// 現状保持しているデータの内、未使用のデータのみを残して、使用済みデータは解放。
// 不足分は新たに読み込む。つまり、_teacherの要素は全て未使用のデータとなる。
// @arg unused_start_idx
//     : このインデックスから末尾までは、潰さずに再利用。
//       (再利用というか、使ってないからまだ捨てない)
// @arg read_threshold
//     : 少なくともこれ以上の教師数を持つ。
template<typename T>
int __FastDataloader::__read_files_sequential(const int unused_start_idx, const int read_threshold) {
	//std::cout << "cinfo: start __read_files_sequential(), _file_names_with_flag.size() = " << _file_names_with_flag.size() << std::endl << std::flush;
	// NOTE: まだファイル一覧が無ければ取得する。
	if (_file_names_with_flag.size() <= 0) {
		//std::cout << "info: _file_names_with_flag.size() <= 0" << std::endl;
		std::vector<std::string> file_names;
		if (!_get_file_list(file_names, _dir)) {
			std::cout << "Error: failed to get file_list, dir = " << _dir << std::endl << std::flush;
			return 0;
		}
		if (file_names.size() <= 0) {
			std::cout << "Error: file_names.size() = " << file_names.size() << " <= 0" << std::endl << std::flush;
			return 0;
		}

		for (const auto& file_name : file_names) {
			_file_names_with_flag.emplace_back(file_name, false);
		}

		if (_shuffle) {
			// HACK
			//     : 本当は、if (_file_names_with_flag.size() <= 0) {} の部分をinit_at_iter()に丸々移してやれば良いんだけど、
			//       __read_files_sequential() よりも先にinit_at_iter() が呼ばれる保証が無いと言えば無いので、念の為このままで。。。
			// NOTE: 今回が初めての読み込みのはずなので、シャッフルをしてやる。
			shuffle(_file_names_with_flag);
		}
	}

	const auto& data = get_data<T>();
	// https://theolizer.com/cpp-school2/cpp-school2-23/
	// メンバ関数内では、thisポインタがコピーキャプチャされる。
	// https://learn.microsoft.com/ja-jp/cpp/error-messages/compiler-errors-2/compiler-error-c3493?view=msvc-170
	// [] の中が空だと、キャプチャしないらしい。
	// https://qiita.com/YukiMiyatake/items/8d10bca26246f4f7a9c8
	// キャプチャリストの中身は、"=", "&", "obj", "&obj" の4種類が許されており、
	// "=", "obj" がコピーキャプチャ、"&", "&obj"が参照キャプチャとなる。
	// NOTE
	//     : ※read_files_all() の場合は、store_teachers 時にidxs_list を渡すので、こちらで管理する必要が無い。
	auto init_idxs = [&]() {
		_idxs = arange<int>(data.size());
		if (_shuffle) {
			shuffle(_idxs);
		}
	};

	//std::cout << "info: __clear_data_except_unused() start" << std::endl;

	// ここで、末尾のまだ使ってない部分を先頭に移す。
	__clear_data_except_unused<T>(unused_start_idx);
	_next_idx = 0;
	//std::cout << "info: __clear_data_except_unused() done" << std::endl;

	// TODO: 規定のサイズを満たすまで読み込む。
	const int n_files = _file_names_with_flag.size();
	int i = 0;
	for (auto&& ff : _file_names_with_flag) {
		++i;
		if (ff.second) {    // 読み込み済み
			//std::cout << "info: pass file[" << i << "/" << n_files << "]" << std::endl;
			continue;
		}

		if (!file_extension_is_ok<T>(ff.first)) {
			std::cout << "Error: " << ff.first << " has an inappropriate extension, so passed reading." << std::endl;
		}
		else {
			std::cout << "info: reading[" << i << "/" << n_files  << "] " << ff.first << std::endl;
			if (!_read_one_file<T>(ff.first)) {
				std::cout << "Error: failed to read, file_name = " << ff.first << std::endl;
				return 0;
			}
			std::cout << "info: data.size() = " << data.size() << std::endl;
			ff.second = true;
		}

		if (data.size() >= read_threshold) {    // 必要な数読み込めた。
			init_idxs();
			return 1;
		}

	}

	// HACK
	//     : ここまで来るのは、基本的には必要な数が読み込めなかった場合。
    //       だけど、時に、minimum_threshold より合計教師数が少なかった場合もここに来る。
	//       なので、そのあたりをわけないといけない。(その場合、minimum_threshold未満しか読み込めないのは仕方ないことなので。)
	//       一旦は1 を返すことに。
	//     -> 期待する数(read_threshold) は読み込めなかったが、batch_size 以上は読み込めたならok
	if (data.size() >= _batch_size) {
		init_idxs();
		return 1;
	}
	else {
		std::cout << "Error: __read_files_sequential() cannot read enough data." << std::endl;
		return 0;
	}
}

// https://qiita.com/i153/items/38f9688a9c80b2cb7da7
// 明示的なインスタンス化
// これが無いと、他のファイルから呼んだ時に、
// 実体が無いためにリンクエラーになる。
// ちなみに、特殊化されたテンプレート関数は、明示的なインスタンス化の必要はない。
// 実装を cpp に置くことで、コンパイル時間の短縮が出来る。
template int __FastDataloader::__read_files_sequential<teacher_t>(const int, const int);
template int __FastDataloader::__read_files_sequential<TrainingData>(const int, const int);
