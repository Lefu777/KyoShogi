#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef NOMINMAX
#include <iostream>
#include <string>

#include "util.hpp"

// HACK: 状態遷移考え方がきれいにはなりそ。
// HACK: 正直まじでこねくり回した書き方しててキモイ。なんでこうなってしまったのか。。。
typedef class StopWatch {
private:
	LARGE_INTEGER _freq, _start;

	// 経過時間[ms]
	// stop or pause が呼ばれるごとに更新される。
	double _elapsed_ms;
	// 時間計測中であるか否か
	bool _is_active;
	// pause 中であるか否か
	bool _is_pausing;

    // 現在実行しているプロセス(処理) を表示する際の名前
	std::string _proc_name;

	// この関数が複数呼ばれてもスレッドセーフ
	// ある期間(最後のstart or resume からこれが呼ばれるまでの間)の経過時間[ms]を計算してstore
	// @arg elapsed_ms[out]: 求めた経過時間をstore
	// @return
	//     : 0: 失敗, 1: 成功
	int _store_elapsed_ms(double& elapsed_ms) {
		LARGE_INTEGER end;
		if (!QueryPerformanceCounter(&end)) {
			return 0;
		}

		elapsed_ms = 1000 * ((double)(end.QuadPart - _start.QuadPart) / _freq.QuadPart);

		if (elapsed_ms < 0) {
			return 0;
		}

		return 1;
	}

	// ある期間(最後のstart or resume からこれが呼ばれるまでの間)の経過時間を、
	// 与えられたそれ以前までの経過時間(prev_elapsed_ms)に足す
	// @arg prev_elapsed_ms
	//     : [in]  "最後のstart or resume からこれが呼ばれるまでの期間"より前の時点での経過時間
	//     : [out] "最後のstart or resume からこれが呼ばれるまでの期間"での経過時間を足した、合計の経過時間
	int _inc_store_elapsed_ms(double& prev_elapsed_ms) {
		double new_elapsed_ms;
		if (!_store_elapsed_ms(new_elapsed_ms)) {
			return 0;
		}
		prev_elapsed_ms += new_elapsed_ms;
		return 1;
	}

	// データ競合を起こすので並列で呼ばないこと(_freq, _start)
	// 今この瞬間を、開始時刻として記録する
	int _record_start_time() {
		if (!QueryPerformanceFrequency(&_freq)) {
			return 0;
		}
		if (!QueryPerformanceCounter(&_start)) {
			return 0;
		}
		return 1;
	}

	// 現在の状態を表す文字列を返す
	std::string _get_state_str() const {
		std::stringstream ss;
		ss 
			<< "[_is_active = " << bts(_is_active) << "]"
			<< "[_is_pausing = " << bts(_is_pausing) << "]"
			<< "[_proc_name = " << _proc_name << "]";
		return ss.str();
	}

public:
	// 以下の使用方法が想定される。
	// start -> stop
	// start -> (pause -> resume)^n -> stop
	// start -> (pause -> resume)^n -> pause -> stop
	// 又、他にも以下の使い方が想定される(追加)。
	// start_pause -> (resume -> pause)^n -> stop
	StopWatch()
		: _elapsed_ms(0), _is_active(false), _is_pausing(false), _proc_name("")
	{
		// https://www.usefullcode.net/2006/12/large_integerularge_integer.html
		// https://www14.big.or.jp/~ken1/tech/tech19.html
		memset(&_freq, 0x00, sizeof(_freq));
		memset(&_start, 0x00, sizeof(_start));
	}

	// データ競合を起こすので並列で呼ばないこと
	// 計測の開始
	// [!a, !p] -> [a, !p]
	// stop     -> start
	// @return
	//     : 0: 失敗, 1: 成功
	int start() {
		if (!(!_is_active && !_is_pausing)) {
			std::cout << "Error: failed to start(), " << _get_state_str() << std::endl;
			return 0;
		}

		_elapsed_ms = 0;
		_is_active ^= 1;

		if (!_record_start_time()) {
			return 0;
		}
		return 1;
	}

	// 計測の開始と同時にpauseする。以下の使い方を想定。
	// start_pause();
	// for () {
	//     resume();
	//     // 何らかの処理
	//     pause();
	// }
	// stop();
	int start_pause() {
		if (!(!_is_active && !_is_pausing)) {
			std::cout << "Error: failed to start_pause(), " << _get_state_str() << std::endl;
			return 0;
		}

		_elapsed_ms = 0;
		_is_pausing ^= 1;

		return 1;
	}

	// 計測を途中から再開
	// [!a, p] -> [a, !p]
	// pause   -> resume
	int resume() {
		if (!(!_is_active && _is_pausing)) {
			std::cout << "Error: failed to resume(), " << _get_state_str() << std::endl;
			//return 0;
			exit(1);
		}

		_is_pausing ^= 1;
		_is_active ^= 1;

		if (!_record_start_time()) {
			return 0;
		}
		return 1;
	}

	// データ競合を起こすので並列で呼ばないこと(メンバ変数である_elapsed_ms に対する加算があるので)
	// 計測の終了(一番最後に呼ぶ)
	// [a, !p] | [a, !p] | [!a, p] -> [!a, !p]
	// (start  | resume  | pause)  -> stop
	// @return
	//     : 0: 失敗, 1: 成功
	int stop() {
		if (
			!((_is_active && !_is_pausing) || (!_is_active && _is_pausing))
		) {
			std::cout << "Error: failed to stop(), " << _get_state_str() << std::endl;
			return 0;
		}


		if (_is_pausing) {    // pause
			_is_pausing ^= 1;
		}
		else {    // start | resume
			_is_active ^= 1;
			if (!_inc_store_elapsed_ms(_elapsed_ms)) {
				return 0;
			}
		}

		return 1;
	}

	// 計測を一時中断して、_elapsed_ms に加算。、
	// [a, !p] | [a, !p] -> [!a, p]
	// (start  | resume) -> pause
	int pause() {
		if (!(_is_active && !_is_pausing)) {
			std::cout << "Error: failed to pause(), " << _get_state_str() << std::endl;
			//return 0;
			exit(1);
		}
		_is_pausing ^= 1;
		_is_active ^= 1;

		if (!_inc_store_elapsed_ms(_elapsed_ms)) {
			return 0;
		}

		return 1;
	}

	int start_with_print(const std::string& proc_name) {
		_proc_name = proc_name;
		if (!start()) {
			return 0;
		}

		std::cout << "[" << _proc_name << "] start" << std::endl;
		return 1;
	}

	int start_pause_with_print(const std::string& proc_name) {
		_proc_name = proc_name;
		if (!start_pause()) {
			return 0;
		}

		std::cout << "[" << _proc_name << "] start" << std::endl;
		return 1;
	}

	int stop_with_print() {
		if (!stop()) {
			return 0;
		}

		std::cout << "[" << _proc_name << "] done, time = " << elapsed_ms() << "ms" << std::endl;
		_proc_name = "";
		return 1;
	}

	// この関数が複数呼ばれてもスレッドセーフ
	// 現在の途中の、暫定の経過時間を返す
	double elapsed_ms_interim() {
		if (!_is_active) {    // 計測時間外に呼び出すと無意味な値になるのでダメ
			std::cout
				<< "Error: failed to get_elapsed_time_interim(), because calling at inappropriate times, " 
			    << _get_state_str() << std::endl;
			return -1;
		}

		double elapsed_ms = _elapsed_ms;
		if (!_inc_store_elapsed_ms(elapsed_ms)) {
			std::cout << "Error: failed to calculate_elapsed_time(), " << _get_state_str() << std::endl;
			return -1;
		}
		return elapsed_ms;
	}

	// 計測後に、計測した時間を返す
	double elapsed_ms() const {
		if (_is_active) {    // 計測時間内に呼び出すと無意味な値になるのでダメ
			std::cout
				<< "Error: failed to get_elapsed_time(), because calling at inappropriate times, "
				<< _get_state_str() << std::endl;
			return -1;
		}
		
		return _elapsed_ms;
	}
} stopwatch_t;