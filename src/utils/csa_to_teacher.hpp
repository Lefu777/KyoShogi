#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_map>

#include "cshogi.h"
#include "parser.h"
#include "dataloader.h"
#include "util.hpp"
#include "types.hpp"


struct PlyAndScore {
	int ply;
	int score;
};

class CsaToTeacher {
private:
	int _minimum_rating;                // 両者minimum_rating 以上の棋譜のみを採用。
	int _minimum_ply;                   // minimum_ply手以上の棋譜のみを採用。
	                                    // 但し、引き分けの棋譜についてはこの限りではない。
	PlyAndScore _evenness_threshold;    // 互角なもの同士のまともな棋譜であれば、
	                                    // ply手以下では常に、評価値の絶対値がscore以下のはず。
	int _minimum_max_eval;               // 評価値の絶対値の最大値は、少なくともminimum_max_eval 以上のはず
	std::vector<teacher_t> _teachers;   // 読み込んだ教師を格納
	int _ponanza;                        // 勝率, 評価値 変換公式におけるponanza定数


public:
	// 採用条件は各ファイルにおいて一定のはずなので、コンストラクタで初期化する。
	CsaToTeacher(
		const int minimum_rating,
		const int minimum_ply,
		const PlyAndScore evenness_threshold,
		const int minimum_max_eval,
		const int ponanza = 600
    )
		: _minimum_rating(minimum_rating), _minimum_ply(minimum_ply), _evenness_threshold(evenness_threshold),
		_minimum_max_eval(minimum_max_eval), _ponanza(ponanza)
	{
	}

	// TODO
	//     : progress 一旦切って、テストする。
	//     : floodgate、ソフト投入してから15戦ぐらいレートつかないけど、その間レートはパーサーのデフォルト値 = 0になっている。
	//       なので、一旦レーティング票を作成してから読んだ方がよさそう。
	//       後、レーティングには変動があるので、最低レーティングを保持するべきかな？
	//     ; もし片方のレーティングが不明でも、片方のレーティングが基準以上で、且つレーティングが不明な方が勝っている場合は採用とする。
	// 一つのディレクトリ内の全てのcsaファイルを変換。
	// @arg src_dir: このディレクトにある.csa ファイルを全て教師化
	// @arg output_base_filename: これを書き出す教師名のbase とする。
	// @arg minimum_write_ply: この手数以上を書き出す。
	// @arg eval_limit: この評価値未満の局面のみを書き出す。
	int csa_to_teacher(
		const std::string& src_dir,
		const std::string& output_base_filename,
		const int& minimum_write_ply,
		const int& eval_limit
    ) {
		std::vector<std::string> filenames;
		if (!__FastDataloader::_get_file_list(filenames, src_dir)) {
			std::cout << "Error: failed to get file_list, src_dir = " << src_dir << std::endl << std::flush;
			return 0;
		}
		if (filenames.size() <= 0) {
			std::cout << "Error: there is no file in \"" << src_dir << "\"" << std::endl << std::flush;
			return 0;
		}

		__Board board;
		Progress prog_ratings("make rating table", filenames.size(), 10);
		std::unordered_map<std::string, int> minimum_rating_table;
		 std::vector<parser::__Parser> parsers;

		for (const auto& filename : filenames) {
			if (filename.substr(filename.size() - 4) != ".csa") {
				std::cout << "Error: " << filename << " is not .csa, so pass reading file." << std::endl;
				continue;
			}
			parser::__Parser psr;
			psr.parse_csa_file(filename);
			parsers.emplace_back(psr);

			// NOTE: rating0 == 0 なら、未登録でない限り更新する必要なし。
			// HACK; ちょっとこのコードはいただけないね。
			// レーティングテーブルに登録
			auto& table_rating0 = minimum_rating_table[psr.names[0]];
			const auto& rating0 = psr.ratings[0];
			if (minimum_rating_table.find(psr.names[0]) == minimum_rating_table.end()) {    // 存在しない(未登録)
				table_rating0 = rating0;
			}
			else if (table_rating0 != 0 && rating0 != 0) {
				table_rating0 = my_min(table_rating0, rating0);
			}
			else if (rating0 != 0) {
				table_rating0 = rating0;
			}

			auto& table_rating1 = minimum_rating_table[psr.names[1]];
			const auto& rating1 = psr.ratings[1];
			if (minimum_rating_table.find(psr.names[1]) == minimum_rating_table.end()) {    // 存在しない(未登録)
				table_rating1 = rating1;
			}
			else if (table_rating1 != 0 && rating1 != 0) {
				table_rating1 = my_min(table_rating1, rating1);
			}
			else if (rating1 != 0) {
				table_rating1 = rating1;
			}
			prog_ratings.step();
		}
		prog_ratings.finalize();

		//Progress prog("convert csa to teacher", filenames.size(), 10);

		for (const auto& psr : parsers) {
			//prog.step();

			// 投了, 千日手, 入玉勝ち宣言 以外での決着はアウト
			if (psr.endgame != "%TORYO"
				&& psr.endgame != "%SENNICHITE"
				&& psr.endgame != "%KACHI"
			) {
				//std::cout << "    info: illegal end" << std::endl;
				continue;
			}

			// レーティンの制限でアウト
			if (my_min(minimum_rating_table[psr.names[0]], minimum_rating_table[psr.names[1]]) < _minimum_rating) {
				//std::cout << "    info: rating[" << minimum_rating_table[psr.names[0]] << ", " << minimum_rating_table[psr.names[1]] << "]" << std::endl;
				continue;
			}

			// 手数の制限でアウト
			if (psr.endgame != "%SENNICHITE" && psr.moves.size() < _minimum_ply) {
				//std::cout << "    info: ply" << std::endl;
				continue;
			}

			// 互角指標の制限でアウト
			int moves_size = psr.moves.size();
			int n_iter_limit = my_min(moves_size, _evenness_threshold.ply);
			bool evenness_is_ok = true;
			for (int i = 0; i < n_iter_limit; ++i) {
				const auto& score = psr.scores[i];
				if (std::abs(score) > _evenness_threshold.score) {
					evenness_is_ok = false;
					break;
				}
			}
			if (!evenness_is_ok) {
				//std::cout << "    info: not even" << std::endl;
				continue;
			}

			// 引き分けじゃないのに、ずっと評価値0 はおかしいのでアウト
			// 非0 の評価値を発見したか否か
			bool found_not_zero[2] = { true, true };
			if (psr.endgame != "%SENNICHITE") {
				found_not_zero[0] = false;
				found_not_zero[1] = false;
				for (int i = 0; i < moves_size; ++i) {
					if (found_not_zero[i % 2] == false) {
						found_not_zero[i % 2] = (psr.scores[i] != 0);
					}
				}
				// TODO: 評価値がおかしい手番だけを不採用にする。
				// ひとまず、片方だけでも評価値がおかしければ不採用に。
				if (found_not_zero[0] == false || found_not_zero[1] == false) {
					//std::cout << "    info: all zero eval" << std::endl;
					continue;
				}

				// 投了で終了してるのに、評価値の絶対値が閾値に達していない場合、
				// この棋譜の勝敗は信用できない
				if (psr.endgame != "%TORYO") {
					int max_abs_score = 0;
					for (int i = 0; i < moves_size; ++i) {
						int score = psr.scores[i];
						max_abs_score = my_max(max_abs_score, std::abs(score));
					}
					if (max_abs_score < _minimum_max_eval) {
						//std::cout << "    info: incomplete" << std::endl;
						continue;
					}
				}
			}

			//// board.set()
			board.set(psr.sfen);
			//// 対極開始局面での手番からみた結果(勝敗) を計算。
			GameResult parser_result = (GameResult)psr.win;
			float result;    // 現在の手番からした結果。初めは初期局面の手番。
			// HACK: resultで使ってる、0, 0.5, 1 、マジックナンバーなんだよなぁ～～～
			// ひとまず、先手番からみた結果(勝敗) を計算
			switch (parser_result) {
			case Draw:
				result = 0.5;
				break;
			case BlackWin:
				result = 1;
				break;
			case WhiteWin:
				result = 0;
				break;
			}
			// 開始時に後手番なら反転する。
			if (board.turn() == White) {
				result = 1 - result;
			}
			std::cout << psr.path << std::endl;
			std::cout << "    psr.win = " << psr.win << ", result = " << result << std::endl;

			// _teachers に格納。

			// _teachers に格納。
			for (int i = 0; i < moves_size; ++i) {
				const int& ply = i + 1;
				const int& move = psr.moves[i];
				int score = psr.scores[i];
				
				// minimum_write_ply 手以上を教師に含める。
				// score != 0 の指し手のみを教師に含める。
				if (ply >= minimum_write_ply && score != 0 && score < eval_limit) {
					// score は先手から見ての評価値なので、
					// 後手の場合反転する。
					score = (board.turn() == Black ? score : -score);

					const std::string& sfen = board.toSFEN();
					const float& winrate = score_to_winrate(score, PONANZA);

#ifdef DEBUG_
					if (100 <= i && i < 104) {
						std::cout << "    sfen = " << sfen << std::endl;
						//std::cout << "    score = " << score << std::endl;
						std::cout << "    winrate = " << winrate << std::endl;
						std::cout << "    result = " << result << std::endl;
					}
#endif

					_teachers.emplace_back(
						sfen, move, i + 1, winrate, result
					);
				}

				result = 1 - result;    // 手番が変わるので反転
				board.push(move);
			}
		}
		//prog.finalize();

		write_teacher(
			_teachers, output_base_filename, -1, true, true
		);

		return 1;
	}

};
