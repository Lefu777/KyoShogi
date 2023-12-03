#pragma once

#include <algorithm>
#include <vector>
#include <stdio.h>
#include "cshogi.h"
#include "types.hpp"
#include "util.hpp"
#include "random_player.hpp"

// TODO
//     : 複数エンジン間での対局。
//       例えば、model path を複数指定したら、指定した数だけエンジンを立ち上げて、
//       総当たり形式で生成してくれる。(対戦者はrandom で選ぶと実装ラクかな。)
//     : 教師の圧縮。HuffmanCodedPosAndEval 使うのはあり。互換性があると嬉しいからね。

// HACK: 現状、クラスの意味があんまりない。
class GensfenRandom {
public:
	GensfenRandom(
		unsigned long long n_gensfens_total,
		unsigned long long save_interval,
		int max_moves_to_draw,
		std::string file_name
	) : _n_gensfens_total(n_gensfens_total), _save_interval(save_interval),
		_max_moves_to_draw(max_moves_to_draw), _file_name(file_name)
	{
		_curr = 0;
		_player = RandomPlayer();
		_board = __Board();
		_board.reset();
	}

	void gen() {
		std::cout << get_now_str() << " : gensfen start" << std::endl << std::flush;
		int n_loops = _n_gensfens_total / _save_interval;
		for (int i = 0; i < n_loops; ++i) {
			std::ostringstream oss;
			oss << "[" << i + 1 << "/" << n_loops << "] gensfen";
			_gen(oss.str(), _save_interval);
		}
		std::cout << "info: gensfenrand done" << std::endl;
	}

private:
	RandomPlayer _player;
	__Board _board;
	int _curr;    // 現在の局面数。
				  // 次に局面を格納すべきidx でもある。 (故に、常に空っぽの箇所を指す。)
	unsigned long long _n_gensfens_total;
	unsigned long long _save_interval;
	int _max_moves_to_draw;
	std::string _file_name;
	//bool _random_name;

	// TODO
	//     : ply手以上の時にのみ書き出す
	//     : 並列実行。一つの共有するqueueなりvector なりの排他ロックを各スレッドが定期的に獲得して、push_back() する。
	//     : 引き分けの判定 & 謎に引き分けになるバグを調査。
	//       各スレッドの排他ロックを獲得する感覚は、乱数で生成すると良いかもね。各スレッドが割りと別々のタイミングでアクセスするようになるはず。
	// n_gensfens にとても小さい値が入ることはまずないだろう、と仮定したうえでの実装。
	// @arg n_gensfens
	//     : 各対局終了毎に総生成局面数を確認して、これを超えていたら終了。
	//       class 本体とは違い、今回生成する局面数であることに注意。
	// @arg max_moves_to_draw
	//     : max_moves_to_draw 引き分けとなる最大手数。
	void _gen(std::string proc_name, unsigned long long n_gensfens) {
		_curr = 0;
		std::vector<teacher_t> teachers(n_gensfens + _max_moves_to_draw - 1);
		Progress prog(proc_name, n_gensfens, 1e5);
		while (_curr < n_gensfens) {
			_board.reset();
			float result = 0.5;
			int start_idx = _curr;
			int ply = 0;
			if (!_player._isready()) {
				std::cout << "Error: isready failed" << std::endl;
			}
			_player._usinewgame();

			while (ply < _max_moves_to_draw) {    // 現在までの手数が_max_moves_to_draw 未満であれば、着手を許可する
				// check
				// NOTE
				//     : _player._go() よりも前に以下のcheck が必要。_go() がそういう前提で実装してあるので。
				//     : 自明ではあるが、以下のif 内のresult は、最後に着手した手番とは反対の手番から見た結果である。
				if (_board.is_game_over()) {
					result = 0;
					break;
				}
				else if (_board.is_nyugyoku()) {
					result = 1;
					break;
				}

				// TODO; ここで、評価値によっては投了させる。
				// think
				int move = _player._go();

				// store
				// result は試合後に纏めて処理
				ply = (_curr - start_idx) + 1;
				teachers[_curr].sfen = _board.toSFEN();    // ここが、random playerの思考時間の2,3倍掛かってる。
				teachers[_curr].move = move;
				teachers[_curr].ply = ply;
				teachers[_curr].value = 0.5;    // dummy

				// next
				++_curr;
				prog.step();
				_board.push(move);
				_player.direct_push(move);

				if (ply == _max_moves_to_draw) {    // 最大手数到達による引き分け
					// HACK; max_moves_to_draw手目が投了 or 入玉宣言でなかった時点で引き分けとする。
					//if (board.is_game_over()) { result = -1; }
					//else if (board.is_nyugyoku()) { result = 1; }
					//else { result = 0; }
					result = 0.5;
					break;
				}
			}

			for (int i = _curr - 1; i >= start_idx; --i) {
				result = 1 - result;    // 反転。(0.5 なら0.5 のまま。)
				teachers[i].result = result;
			}
		}
		prog.finalize();
		// TODO: ここで局面のshuffle をしたい。
		// write .bin
		teachers.resize(n_gensfens);    // 末尾を切り落とす

		write_teacher(teachers, _file_name, my_min(n_gensfens / 10, 1e6), true, true);
	}
};